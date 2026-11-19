#!/usr/bin/env python3
"""Render a 16-bit relief heightmap as a lit, shadowed, tilted 3D view.

Usage: render_preview.py heightmap16.png out.png --thickness 2.0 [--grid 0.01]
The heightmap is the one written by make_relief.py (white = full thickness).
"""
import argparse
import numpy as np, cv2
from PIL import Image


def shadows(z, px, az_deg, el_deg, soft=3):
    """Fraction of light reaching each cell from a directional light (ray march)."""
    az, el = np.radians(az_deg), np.radians(el_deg)
    dx, dy = np.cos(az), -np.sin(az)            # step towards the light, image coords
    rise = np.tan(el) * px                      # height gained per pixel step
    h, w = z.shape
    lit = np.ones_like(z, np.float32)
    n = int((z.max() - z.min()) / rise) + 2
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    for k in range(1, n, 1):
        zz = cv2.remap(z, (xs + dx * k).astype(np.float32), (ys + dy * k).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        lit = np.minimum(lit, np.clip(1 - (zz - (z + k * rise)) / (0.02 + 0.004 * k), 0, 1))
    return cv2.GaussianBlur(lit, (0, 0), soft)


def shade(z, px, az_deg=135, el_deg=38):
    gy, gx = np.gradient(z, px)
    n = np.stack([-gx, gy, np.ones_like(z)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    az, el = np.radians(az_deg), np.radians(el_deg)
    l = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    diff = np.clip(n @ l, 0, 1)
    lit = shadows(z.astype(np.float32), px, az_deg, el_deg)
    # ambient occlusion: how far a cell sits below its surroundings
    ao = np.clip(1 - 1.6 * np.clip(cv2.GaussianBlur(z, (0, 0), 0.12 / px) - z, 0, None), 0.45, 1)
    return (0.30 * ao + 0.78 * diff * (0.25 + 0.75 * lit)) * (0.9 + 0.1 * ao)


def tilt_view(z, col, px, tilt_deg, yaw_deg, out_w, table=(58, 60, 66)):
    """Oblique orthographic view: yaw the slab in-plane, then tilt the camera."""
    h, w = z.shape
    pad = int(0.45 * max(h, w))
    M = cv2.getRotationMatrix2D((w / 2, h / 2), yaw_deg, 1.0); M[:, 2] += pad
    size = (w + 2 * pad, h + 2 * pad)
    zr = cv2.warpAffine(z, M, size, flags=cv2.INTER_LINEAR, borderValue=-1.0)
    cr = cv2.warpAffine(col, M, size, flags=cv2.INTER_LINEAR, borderValue=table)
    inside = zr >= 0
    # table gets a soft contact shadow from the slab
    sh = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), 0.25 / px)
    cr = np.where(inside[..., None], cr, np.array(table, np.float32) * (1 - 0.55 * sh[..., None]))
    zr = np.where(inside, zr, 0)
    t = np.radians(tilt_deg); c, s = np.cos(t), np.sin(t)
    H, W = zr.shape
    sy = np.arange(H)[:, None] * c - zr / px * s          # screen row of every cell
    top = int(np.floor(sy.min())) - 2
    out_h = int(np.ceil(sy.max())) - top + 4
    img = np.zeros((out_h, W, 3), np.float32); img[:] = table
    horizon = np.full(W, out_h, np.int32)                  # lowest row already drawn, per column
    cols = np.arange(W)
    for y in range(H - 1, -1, -1):                         # near to far, keeping the nearest
        r = np.clip(np.floor(sy[y] - top).astype(np.int32), 0, out_h)
        draw = r < horizon
        if not draw.any():
            continue
        # wall darkening for the stretched (vertical) part of the span
        for x in cols[draw]:
            span = horizon[x] - r[x]
            img[r[x]:horizon[x], x] = cr[y, x] * (0.62 if span > 3 else 1.0)
        horizon = np.where(draw, r, horizon)
    # crop to the projected slab plus a margin
    rows = (sy - top)[inside]; xs = np.where(inside.any(0))[0]
    m = int(0.06 * W)
    img = img[max(int(rows.min()) - m, 0):int(rows.max()) + m, max(xs.min() - m, 0):xs.max() + m]
    k = out_w / img.shape[1]
    return cv2.resize(img, (out_w, int(img.shape[0] * k)), interpolation=cv2.INTER_AREA)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("heightmap"); p.add_argument("out")
    p.add_argument("--thickness", type=float, required=True, help="inches at white")
    p.add_argument("--grid", type=float, default=0.01, help="inches per pixel")
    p.add_argument("--tilt", type=float, default=40); p.add_argument("--yaw", type=float, default=-14)
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--wood", default="196,154,108", help="R,G,B of the material")
    a = p.parse_args()
    z = np.array(Image.open(a.heightmap)).astype(np.float32) / 65535 * a.thickness
    k = 2  # render at twice the heightmap resolution so walls stay smooth
    z = cv2.resize(z, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC); px = a.grid / k
    wood = np.array([float(v) for v in a.wood.split(",")], np.float32)
    col = np.clip(shade(z, px)[..., None] * wood, 0, 255).astype(np.float32)
    img = tilt_view(z, col, px, a.tilt, a.yaw, a.width)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(a.out)
    Image.fromarray(np.clip(cv2.resize(col, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), 0, 255).astype(np.uint8)).save(a.out.replace(".png", "_top.png"))


if __name__ == "__main__":
    main()
