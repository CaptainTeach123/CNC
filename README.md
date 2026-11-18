# CNC — AltMill 4x4 project files

Design sources, exports and G-code for parts cut on a Sienci Labs AltMill 4x4.

## Layout

| Path | Contents |
| --- | --- |
| `docs/` | Notes and reference (Fusion free-tier limits, workflow) |
| `machine/` | AltMill 4x4 machine profile, Fusion/gSender setup notes |
| `projects/<name>/` | One folder per item to make (copy `projects/_template`) |
| `tools/` | Tool library (bits, feeds and speeds) |

Each project folder:

| Folder | Contents |
| --- | --- |
| `reference/` | Photos, sketches and measurements the design is based on |
| `fusion/` | Fusion Python scripts that build the model, plus `.f3d` archives exported from Fusion |
| `exports/` | STEP / DXF / STL exchanged with other tools |
| `gcode/` | Posted `.nc` / `.gcode` files, named `<part>_<op>_<bit>.nc` |

## Workflow

1. Add photos and **measured dimensions** to `projects/<name>/reference/`.
2. A Fusion script in `fusion/` builds the parametric model (run in Fusion via
   *Utilities → Add-Ins → Scripts and Add-Ins*).
3. Set up toolpaths in Fusion's Manufacture workspace, post with the Grbl post.
4. Save the G-code into `gcode/`, then run it with gSender.

See [docs/fusion-personal-limits.md](docs/fusion-personal-limits.md) and
[machine/altmill-4x4.md](machine/altmill-4x4.md).
