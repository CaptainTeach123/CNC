#!/usr/bin/env python3
"""Turn a picture into a carvable relief (heightmap PNG + STL).

Pipeline: crop to the target aspect -> mask the bright lettering -> fill the
lettering from a text-free copy of the artwork (--art, aligned automatically)
or by inpainting -> optional 2x super-resolution (--sr-model) -> monocular
depth estimation (Depth Anything V2, ONNX) as one global pass plus zoomed-in
tile passes for detail -> compress and sharpen into a bas-relief -> add the
lettering back as a raised, drafted layer (unless --no-text) -> triangulate
onto a solid base, optionally decimated.

Needs: numpy, pillow, opencv-python-headless, onnxruntime, fast-simplification
(for --keep), and the model files
  https://huggingface.co/onnx-community/depth-anything-v2-large/resolve/main/onnx/model.onnx
  https://huggingface.co/Xenova/swin2SR-realworld-sr-x4-64-bsrgan-psnr/resolve/main/onnx/model.onnx
(Depth Anything V2 Large is licensed CC-BY-NC-4.0: non-commercial use only.)
"""
import argparse, struct
from pathlib import Path
import numpy as np, cv2
from PIL import Image

IN = 25.4  # mm per inch


def text_mask(rgb):
    """Near-white, unsaturated, letter-sized blobs."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[..., 1], hsv[..., 2]
    core = ((v > 228) & (s < 25)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, 8)
    keep = np.isin(lab, [i for i in range(1, n) if stats[i, 4] > 400]).astype(np.uint8)
    grow = cv2.dilate(keep, np.ones((9, 9), np.uint8))
    m = (((grow > 0) & (v > 150) & (s < 40)) | (keep > 0)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def register_art(cover, art, tm):
    """Warp a text-free copy of the artwork onto `cover` (SIFT + homography)
    and colour-match it. Returns the warped art and where it is valid."""
    grow = lambda k: cv2.dilate(tm, np.ones((k, k), np.uint8))
    gray = lambda x: cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(8000)
    k1, d1 = sift.detectAndCompute(gray(art), None)
    k2, d2 = sift.detectAndCompute(gray(cover), (grow(21) == 0).astype(np.uint8) * 255)
    good = [m for m, n in cv2.BFMatcher().knnMatch(d1, d2, k=2) if m.distance < 0.7 * n.distance]
    p1 = np.float32([k1[g.queryIdx].pt for g in good]); p2 = np.float32([k2[g.trainIdx].pt for g in good])
    H, inl = cv2.findHomography(p1, p2, cv2.RANSAC, 2.0)
    print(f"art registration: {int(inl.sum())} inliers of {len(good)} matches, scale {H[0, 0]:.3f}")
    h, w = cover.shape[:2]
    warp = cv2.warpPerspective(art, H, (w, h), flags=cv2.INTER_LANCZOS4).astype(np.float32)
    valid = cv2.warpPerspective(np.full(art.shape[:2], 255, np.uint8), H, (w, h)) > 250
    ok = (grow(41) == 0) & valid
    for c in range(3):
        a, b = np.polyfit(warp[..., c][ok][::7], cover[..., c][ok][::7].astype(np.float32), 1)
        warp[..., c] = warp[..., c] * a + b
    return warp, valid


def text_mask_from_art(cover, warp, valid, tm):
    """Lettering = where the cover is clearly brighter than the text-free art.
    Catches hairline strokes that a plain brightness threshold misses."""
    gray = lambda x: cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    diff = cv2.GaussianBlur(gray(cover) - gray(warp), (0, 0), 0.7)
    strong, weak = (diff > 70) & valid, (diff > 38) & valid
    n, lab = cv2.connectedComponents(weak.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool); keep[np.unique(lab[strong])] = True; keep[0] = False
    m = keep[lab]
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    m = np.isin(lab, [i for i in range(1, n) if st[i, 4] > 150])
    # the brightness mask still wins where white lettering sits on a light background
    return (m | (tm > 0)).astype(np.uint8)


def fill_from_art(cover, warp, valid, tm):
    """Replace the lettering (and its glow) in `cover` with the registered art."""
    grow = lambda k: cv2.dilate(tm, np.ones((k, k), np.uint8))
    m = np.clip(cv2.GaussianBlur(grow(51).astype(np.float32), (0, 0), 8) * 1.6, 0, 1)
    m *= cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8))
    out = np.clip(cover * (1 - m[..., None]) + warp * m[..., None], 0, 255).astype(np.uint8)
    hole = ((tm > 0) & (m < 1)).astype(np.uint8)
    if hole.any():
        out = cv2.inpaint(out, cv2.dilate(hole, np.ones((13, 13), np.uint8)), 9, cv2.INPAINT_TELEA)
    return out


def super_resolve_2x(rgb, model, tile=256, overlap=32):
    """Run a 4x super-resolution model over overlapping tiles and keep a 2x result."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    h, w = rgb.shape[:2]
    acc = np.zeros((2 * h, 2 * w, 3), np.float32); wsum = np.zeros((2 * h, 2 * w, 1), np.float32)
    ramp = np.minimum(np.arange(1, 2 * tile + 1), np.arange(2 * tile, 0, -1)).clip(max=2 * overlap) / (2 * overlap)
    win = np.outer(ramp, ramp).astype(np.float32)[..., None]
    starts = lambda n: sorted({min(s, n - tile) for s in range(0, n, tile - overlap)})
    for y in starts(h):
        for x in starts(w):
            inp = (rgb[y:y + tile, x:x + tile].astype(np.float32) / 255).transpose(2, 0, 1)[None]
            out = np.clip(sess.run(None, {name: inp})[0][0].transpose(1, 2, 0), 0, 1)
            out = cv2.resize(out, (2 * tile, 2 * tile), interpolation=cv2.INTER_AREA)
            acc[2 * y:2 * (y + tile), 2 * x:2 * (x + tile)] += win * out
            wsum[2 * y:2 * (y + tile), 2 * x:2 * (x + tile)] += win
        print(f"  super-resolution row y={y} done", flush=True)
    return (acc / wsum * 255).round().astype(np.uint8)


def _infer(sess, rgb, W, H):
    x = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
    x = ((x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).transpose(2, 0, 1)[None]
    d = sess.run(None, {sess.get_inputs()[0].name: x.astype(np.float32)})[0].squeeze()
    return cv2.resize(d, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)


def global_depth(sess, rgb, long_side):
    h, w = rgb.shape[:2]
    k = long_side / max(h, w)
    return _infer(sess, rgb, int(round(w * k / 14)) * 14, int(round(h * k / 14)) * 14).astype(np.float64)


def add_tile_detail(sess, rgb, d, tile, tile_res, hp_sigma):
    """Replace the fine detail of depth map d with detail from zoomed-in tiles.

    Each overlapping tile is run through the model enlarged, matched to d with
    a least-squares scale/shift, and only its high-frequency part is blended
    back in. Tiles that are all backdrop are skipped. Larger value = nearer.
    """
    h, w = d.shape
    hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), hp_sigma)
    flat = 0.03 * (d.max() - d.min())
    acc, wsum = np.zeros_like(d), np.zeros_like(d)
    win1 = np.hanning(tile + 2)[1:-1]
    win = np.outer(win1, win1) + 1e-3
    starts = lambda n: sorted({min(s, n - tile) for s in range(0, n - tile // 2, tile // 2)})
    for y in starts(h):
        for x in starts(w):
            g = d[y:y + tile, x:x + tile]
            if np.ptp(g) < flat:
                fit = g
            else:
                t = _infer(sess, rgb[y:y + tile, x:x + tile], tile_res, tile_res).astype(np.float64)
                A = np.stack([t.ravel(), np.ones(t.size)], 1)
                a, b = np.linalg.lstsq(A, g.ravel(), rcond=None)[0]
                fit = max(a, 0) * t + b
            acc[y:y + tile, x:x + tile] += win * hp(fit)
            wsum[y:y + tile, x:x + tile] += win
        print(f"  tile row y={y} done", flush=True)
    return d - hp(d) + acc / wsum


def guard_detail(prev, new, limit):
    """Accept a detail pass only where it makes small changes. Small tiles lose
    context and can flatten thin structures or shift silhouettes; those show up
    as large changes, so there the previous level is kept."""
    rng = np.percentile(prev, 99.8) - prev.min()
    delta = (new - prev) / rng
    w = np.clip(1 - (np.abs(delta) - limit) / limit, 0, 1).astype(np.float32)
    w = cv2.GaussianBlur(cv2.erode(w, np.ones((9, 9), np.uint8)), (0, 0), 4)
    return prev + w * (new - prev)


def estimate_depth(rgb, model, levels, ppi, cache, guard=0.02):
    """Global pass, then one detail pass per (tile px, model px, blend inches) level.
    Every stage is cached next to `cache` so reruns only compute what is missing.
    The third and finer passes are guarded (see guard_detail)."""
    sess = None
    def session():
        nonlocal sess
        if sess is None:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        return sess
    d = None
    for i, lv in enumerate([None] + levels):
        f = cache.with_suffix(f".L{i}.npy") if cache else None
        if f and f.exists():
            new = np.load(f)
        else:
            print(f"depth level {i}", flush=True)
            new = global_depth(session(), rgb, 1554) if lv is None else add_tile_detail(session(), rgb, d, lv[0], lv[1], lv[2] * ppi)
            if f:
                np.save(f, new)
        d = guard_detail(d, new, guard) if i >= 3 and guard > 0 else new
    return d


def decimate(tris, keep):
    """Quadric decimation: keeps triangles where the detail is, drops them on flats."""
    import fast_simplification
    v, inv = np.unique(tris.reshape(-1, 3), axis=0, return_inverse=True)
    pts, faces = fast_simplification.simplify(v, inv.reshape(-1, 3), target_reduction=1 - keep)
    return pts[faces].astype(np.float32)


def write_stl(path, z, dx, dy, keep=1.0):
    """Heightfield z[row, col] (row 0 = top of picture) on a flat base at Z=0."""
    ny, nx = z.shape
    xs = np.arange(nx) * dx
    ys = (ny - 1 - np.arange(ny)) * dy
    X, Y = np.meshgrid(xs, ys)
    top = np.stack([X, Y, z], -1).astype(np.float32)
    a, b, c, d = top[:-1, :-1], top[:-1, 1:], top[1:, :-1], top[1:, 1:]
    tris = [np.stack([a, c, b], -2).reshape(-1, 3, 3), np.stack([b, c, d], -2).reshape(-1, 3, 3)]
    # perimeter, counter-clockwise seen from above
    ring = np.concatenate([top[-1, :-1], top[:-1, -1][::-1], top[0, 1:][::-1], top[1:, 0]])
    nxt = np.roll(ring, -1, 0)
    r0, n0 = ring.copy(), nxt.copy()
    r0[:, 2] = 0; n0[:, 2] = 0
    tris += [np.stack([ring, r0, n0], 1), np.stack([ring, n0, nxt], 1)]          # walls
    ctr = np.array([xs.mean(), ys.mean(), 0], np.float32)
    tris.append(np.stack([np.broadcast_to(ctr, r0.shape), n0, r0], 1))           # bottom fan
    t = np.concatenate(tris).astype(np.float32)
    if keep < 1:
        t = decimate(t, keep)
    nrm = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(t), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = nrm, t
    with open(path, "wb") as f:
        f.write(b"relief".ljust(80, b"\0")); f.write(struct.pack("<I", len(t))); f.write(rec.tobytes())
    return len(t)


def shade(z, px):
    gy, gx = np.gradient(z, px)
    n = np.stack([-gx, -gy, np.ones_like(z)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    l = np.array([-0.5, -0.6, 0.62]); l /= np.linalg.norm(l)
    return (np.clip(n @ l, 0, 1) ** 0.9 * 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image"); p.add_argument("outdir"); p.add_argument("--model", required=True)
    p.add_argument("--detail-levels", type=int, default=2, help="tiled detail passes: 0 to 3")
    p.add_argument("--detail-guard", type=float, default=0.02, help="largest change (fraction of depth range) the finest pass may make; 0 = off")
    p.add_argument("--art", help="text-free copy of the same artwork, used to fill in under the lettering")
    p.add_argument("--sr-model", help="4x super-resolution ONNX model; doubles the working resolution")
    p.add_argument("--no-text", action="store_true", help="leave the lettering out")
    p.add_argument("--keep", type=float, default=1.0, help="fraction of triangles kept by adaptive decimation")
    p.add_argument("--cache", help="npy path prefix to store/reuse the depth stages")
    p.add_argument("--name", default="relief")
    p.add_argument("--width", type=float, default=6.0, help="inches")
    p.add_argument("--height", type=float, default=9.0, help="inches")
    p.add_argument("--thickness", type=float, default=1.0, help="total stock thickness, inches")
    p.add_argument("--base", type=float, default=0.30, help="solid backing under the deepest cut, inches")
    p.add_argument("--text-raise", type=float, default=0.08, help="lettering height above the scene, inches")
    p.add_argument("--text-draft", type=float, default=0.02, help="width of the sloped letter edge, inches")
    p.add_argument("--text-grow", type=float, default=0.0, help="thicken letter strokes by this much per side, inches")
    p.add_argument("--grid", type=float, default=0.0125, help="mesh spacing, inches")
    p.add_argument("--knee", type=float, default=0.62, help="depth above this is compressed (foreground)")
    p.add_argument("--backdrop", type=float, default=0.15, help="depth fraction below this is flat background")
    p.add_argument("--detail", type=float, default=0.9, help="local-relief boost")
    p.add_argument("--texture", type=float, default=0.012, help="surface texture from image brightness, inches")
    a = p.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    rgb = np.array(Image.open(a.image).convert("RGB"))
    h, w = rgb.shape[:2]
    asp = a.width / a.height
    if w / h > asp:
        nw = int(round(h * asp)); o = (w - nw) // 2; rgb = rgb[:, o:o + nw]
    else:
        nh = int(round(w / asp)); o = (h - nh) // 2; rgb = rgb[o:o + nh]
    h, w = rgb.shape[:2]
    ppi = w / a.width

    cache = Path(a.cache) if a.cache else None
    stage = lambda tag: cache.with_suffix(tag) if cache else None

    tm = text_mask(rgb)
    if a.art:
        warp, valid = register_art(rgb, np.array(Image.open(a.art).convert("RGB")), tm)
    f = stage(".clean.png")
    if f and f.exists():
        clean = np.array(Image.open(f).convert("RGB"))
    else:
        if a.art:
            clean = fill_from_art(rgb, warp, valid, tm)
        else:
            clean = cv2.inpaint(rgb, cv2.dilate(tm, np.ones((13, 13), np.uint8)), 9, cv2.INPAINT_TELEA)
        if a.sr_model:
            clean = super_resolve_2x(clean, a.sr_model)
        if f:
            Image.fromarray(clean).save(f)
    if a.art:  # sharper lettering outline for the raised text layer
        tm = text_mask_from_art(rgb, warp, valid, tm)

    # everything from here on runs at the working resolution of `clean`
    if clean.shape[1] != w:
        k = clean.shape[1] / w
        tm = (cv2.resize(cv2.GaussianBlur(tm.astype(np.float32), (0, 0), 0.8), (clean.shape[1], clean.shape[0]),
                         interpolation=cv2.INTER_CUBIC) > 0.5).astype(np.uint8)
        h, w = clean.shape[:2]
        ppi = w / a.width

    # detail levels: (tile size as a fraction of image width, model input px, blend radius in inches at 6 in wide)
    levels = [(0.43, 1036, 0.07), (0.215, 1036, 0.03), (0.107, 770, 0.014)][:a.detail_levels]
    levels = [(int(fr * w) // 2 * 2, res, sig * a.width / 6.0) for fr, res, sig in levels]
    d = estimate_depth(clean, a.model, levels, ppi, cache, a.detail_guard)

    # --- scene: normalise, flatten the backdrop, compress the near foreground
    hi = np.percentile(d, 99.8)
    lo = d.min() + a.backdrop * (hi - d.min())
    s = np.clip((d - lo) / (hi - lo), 0, 1)
    s = np.where(s > a.knee, a.knee + (s - a.knee) * 0.45, s)
    s /= s.max()
    # local-relief boost (unsharp mask at two scales), kept off the backdrop
    # smoothing radii below are quoted for a 6 in wide piece and scale with size
    sc = a.width / 6.0
    g = lambda x, inch: cv2.GaussianBlur(x, (0, 0), inch * sc * ppi)
    fg = np.clip(s / 0.08, 0, 1)
    s = s + a.detail * fg * (0.6 * (s - g(s, 0.25)) + 0.8 * (s - g(s, 0.06)))
    s = np.clip(s, 0, None); s /= np.percentile(s, 99.95); s = np.clip(s, 0, 1)

    relief = a.thickness - a.base - a.text_raise
    z = s * relief
    # fine texture (feathers, hair, folds) from brightness, band-passed
    lum = cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255
    bp = g(lum, 0.006) - g(lum, 0.03)
    z += a.texture * fg * np.clip(bp / 0.08, -1, 1)
    z = np.clip(z, 0, relief)

    # --- lettering: rides on a smoothed copy of the scene, with drafted edges
    if a.text_grow > 0:  # fatten hairline strokes so they survive the cutter
        k = 2 * int(round(a.text_grow * ppi)) + 1
        tm = cv2.dilate(tm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    dist = cv2.distanceTransform(tm, cv2.DIST_L2, 5) / ppi
    ramp = np.clip(dist / a.text_draft, 0, 1)
    k = int(0.08 * sc * ppi) | 1
    carrier = g(cv2.dilate(z, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), 0.08)
    ztext = carrier + a.text_raise * ramp
    if not a.no_text:
        z = np.where(tm > 0, np.maximum(z, ztext), z)
    z = g(z, 0.004) + a.base

    # --- sample onto the mesh grid
    nx, ny = int(round(a.width / a.grid)) + 1, int(round(a.height / a.grid)) + 1
    zg = cv2.resize(z, (nx, ny), interpolation=cv2.INTER_AREA)
    zg = np.clip(zg, a.base, a.thickness)
    n = write_stl(out / f"{a.name}.stl", zg * IN, a.width * IN / (nx - 1), a.height * IN / (ny - 1), a.keep)
    Image.fromarray(np.round(zg / a.thickness * 65535).astype(np.uint16)).save(out / f"{a.name}_heightmap16.png")
    Image.fromarray(shade(z, 1 / ppi)).resize((1304, int(1304 * h / w)), Image.LANCZOS).save(out / f"{a.name}_preview.png")
    print(f"{nx}x{ny} grid, {n} triangles, Z {zg.min():.3f}..{zg.max():.3f} in, "
          f"size {a.width}x{a.height}x{a.thickness} in (STL in mm)")


if __name__ == "__main__":
    main()
