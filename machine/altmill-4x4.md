# AltMill 4x4 machine profile

Values from retailer/partner listings (Nov 2026) — **verify against your own
machine and Sienci's spec page** before relying on them for clearances.

| Item | Value |
| --- | --- |
| Working area (X × Y) | approx. 49 × 49 in (≈1245 × 1245 mm) |
| Max Z height | approx. 5.5 in (≈140 mm) |
| Motion | Ball screws + HGR15 linear guides, closed-loop steppers |
| Rapids | ~10,000 mm/min (≈400 IPM) |
| Controller | Sienci SLB-EXT running grblHAL |
| Spindle mount | 80 mm (spindle bought separately) |
| Sender | gSender (connect as grblHAL) |

## To fill in

- [ ] Spindle model, power, RPM range
- [ ] Collet sizes on hand
- [ ] Wasteboard / clamping system and usable Z under the gantry
- [ ] Dust shoe clearance
- [ ] Touch plate / tool length sensor

## Fusion setup

- Post processor: stock **Grbl** post (`grbl.cps`); grblHAL accepts its output.
- Safe retracts: **Clearance Height** rather than G28, so an un-homed machine
  does not make an unexpected move.
- Units: pick one per project and state it in the project README.
- WCS origin: note it in each project (e.g. front-left corner, top of stock).
