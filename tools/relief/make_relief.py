#!/usr/bin/env python3
"""Turn a picture into a carvable relief (heightmap PNG + watertight STL).

Pipeline: crop to the target aspect -> mask the bright lettering -> inpaint it
away -> monocular depth estimation (Depth Anything V2, ONNX) -> compress and
sharpen into a bas-relief -> add the lettering back as a raised, drafted layer
-> triangulate onto a solid base.

Needs: numpy, pillow, opencv-python-headless, onnxruntime, and the model file
  https://huggingface.co/onnx-community/depth-anything-v2-large/resolve/main/onnx/model.onnx
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


def estimate_depth(rgb, model, levels, ppi, cache):
    """Global pass, then one detail pass per (tile px, model px, blend inches) level.
    Every stage is cached next to `cache` so reruns only compute what is missing."""
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
            d = np.load(f); continue
        print(f"depth level {i}", flush=True)
        d = global_depth(session(), rgb, 1554) if lv is None else add_tile_detail(session(), rgb, d, lv[0], lv[1], lv[2] * ppi)
        if f:
            np.save(f, d)
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
    p.add_argument("--detail-levels", type=int, default=2, help="tiled detail passes: 0, 1 or 2")
    p.add_argument("--upscale", type=float, default=1.0, help="working-resolution multiplier")
    p.add_argument("--keep", type=float, default=1.0, help="fraction of triangles kept by adaptive decimation")
    p.add_argument("--cache", help="npy path prefix to store/reuse the depth stages")
    p.add_argument("--name", default="relief")
    p.add_argument("--width", type=float, default=6.0, help="inches")
    p.add_argument("--height", type=float, default=9.0, help="inches")
    p.add_argument("--thickness", type=float, default=1.0, help="total stock thickness, inches")
    p.add_argument("--base", type=float, default=0.30, help="solid backing under the deepest cut, inches")
    p.add_argument("--text-raise", type=float, default=0.08, help="lettering height above the scene, inches")
    p.add_argument("--text-draft", type=float, default=0.02, help="width of the sloped letter edge, inches")
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

    tm = text_mask(rgb)
    clean = cv2.inpaint(rgb, cv2.dilate(tm, np.ones((13, 13), np.uint8)), 9, cv2.INPAINT_TELEA)
    levels = [(560, 1036, 0.07), (280, 770, 0.03)][:a.detail_levels]
    d = estimate_depth(clean, a.model, levels, ppi, Path(a.cache) if a.cache else None)

    # work at a finer pixel pitch so edges stay smooth on larger pieces
    if a.upscale > 1:
        W, H = int(w * a.upscale), int(h * a.upscale)
        up = lambda x: cv2.resize(x, (W, H), interpolation=cv2.INTER_CUBIC)
        d, clean = up(d), cv2.resize(clean, (W, H), interpolation=cv2.INTER_LANCZOS4)
        tm = (up(cv2.GaussianBlur(tm.astype(np.float32), (0, 0), 0.8)) > 0.5).astype(np.uint8)
        h, w, ppi = H, W, W / a.width

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
    dist = cv2.distanceTransform(tm, cv2.DIST_L2, 5) / ppi
    ramp = np.clip(dist / a.text_draft, 0, 1)
    k = int(0.08 * sc * ppi) | 1
    carrier = g(cv2.dilate(z, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))), 0.08)
    ztext = carrier + a.text_raise * ramp
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
