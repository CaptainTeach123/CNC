# The Justice of Kings — book cover relief

3D relief carving of the book cover. Personal, non-commercial piece: the
artwork is copyrighted, so the source image is not stored in this repo.

- **Finished size:** 12 × 18 × 1 in (304.8 × 457.2 × 25.4 mm)
- **Units:** STL is in **millimetres**
- **WCS origin:** front-left corner, **bottom** of stock (STL Z=0 is the back face)
- **Material:** to be decided (tight-grained hardwood holds the lettering best)

![preview](exports/justice_of_kings_preview.png)

## Files

| File | What it is |
| --- | --- |
| `exports/justice_of_kings.stl` | Watertight relief mesh, ~1.3M triangles (4.3M-triangle 0.010 in grid, adaptively decimated) |
| `exports/justice_of_kings_heightmap16.png` | 16-bit heightmap, 1201 × 1801 px at 0.010 in per pixel (white = 1.000 in, black = 0) |
| `exports/justice_of_kings_preview.png` | Shaded preview |

An earlier 6 × 9 in version is in the git history and can be regenerated with
`--width 6 --height 9`.

## Heights (all estimated from the picture)

| Feature | Height above back face |
| --- | --- |
| Flat backdrop | 0.20 in (deepest cut is 0.80 in from the top) |
| Statue / wings / wolves | ≈0.32–0.55 in |
| Seated figure | ≈0.55–0.72 in |
| Dais (foreground) | ≈0.72–0.85 in |
| Lettering | 0.11 in above the scene under it, 0.03 in drafted edges, max 0.97 in |

Thinnest letter strokes are about 0.08 in wide ("RICHARD SWAN") and 0.11 in
(title). The relief is the same depth as the 6 × 9 version but spread over
twice the width, so it is proportionally shallower; thicker stock
(`--thickness 1.5` or `2`) restores the depth.

## Regenerating / adjusting

```bash
python3 tools/relief/make_relief.py <cover image> projects/justice-of-kings-relief/exports \
  --model <depth-anything-v2-large model.onnx> --name justice_of_kings --cache depth.npy \
  --width 12 --height 18 --base 0.2 --text-raise 0.11 --text-draft 0.03 \
  --texture 0.02 --upscale 2 --grid 0.01 --keep 0.3
```

The depth estimate takes about 25 minutes on 4 CPU cores: one overall pass, 24
zoomed-in tiles, then about 100 smaller tiles for the finest detail. `--cache`
saves each stage so later tweaks take about a minute.

Useful knobs: `--base` (backing thickness; smaller = deeper relief),
`--text-raise`, `--detail`, `--texture`, `--knee` (how much the foreground
dais is flattened), `--grid` (mesh resolution), `--keep` (fraction of triangles kept).

## Carving in Fusion (personal licence)

1. **Insert → Insert Mesh**, pick the STL, units **Millimeter**.
2. Manufacture workspace → **New Setup**: stock = fixed box 304.8 × 457.2 × 25.4 mm,
   model = the mesh body, origin as above (or top of stock if you prefer to
   zero on the surface).
3. **Roughing — 3D Adaptive Clearing**, 1/4 in flat end mill, leave 0.02 in
   radial and axial stock.
4. **Finishing — 3D Parallel**, tapered ball nose (1/32 in tip radius or
   smaller, ≥ 0.85 in reach), stepover about 0.005–0.008 in. Expect several
   hours. A second pass at 90° cleans up the lettering.
5. Post each operation as its own file with the Grbl post (no automatic tool
   change on the free licence) into `gcode/`.

Not yet verified on the machine: that the personal licence runs 3D Adaptive /
Parallel directly on a mesh body in your install. If it refuses, convert the
mesh first (Mesh → Convert Mesh) or use the heightmap in another CAM tool.

**Workholding:** a blank cut exactly 12 × 18 in leaves nothing to clamp. Use an
oversize blank and cut the outline last (with tabs), or hold it with tape and
CA glue.
