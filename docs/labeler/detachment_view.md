# Detachment default view (plan D.B)

The owner's 2026-10-15 request: the detachment default needs Afrac lower and upper,
Prad,div lower and upper, the gas total, the IRTV temperature, and 2-D views of EFIT and
the TangTV inversion; not the individual bolometer and D-alpha chords. IRTV holds no
calibrated temperature trace, so the IRTV row is the heat-flux map (its temperature twin
is noted below); Afrac upper has no source.

## Rows (`src/labeler/events/panels/detachment.py`, store `PANEL_VERSIONS` 9)

In this order; a row whose source is not verified is left out, never guessed.

| Row | Source | Units |
|---|---|---|
| Afrac lower | the labeler's Afrac bins, `round4/detach/bins/<shot>.npz` (`indicator_path`): an uncalibrated local Jsat ratio at the lower outer strike point, not the published Eldon Afrac (caveat on the row). Line 1: the producer's valid bins. Line 2 ("ungated, invalid bins"): on the bins the producer marks invalid, the same ratio with its gates removed (below) | dimensionless, step bins |
| Afrac upper | **left out (owner, 2026-10-16)**: no node, PTDATA pointname or LANGMUIR leaf found | |
| Prad,div lower | `\BOLOM::PRAD_DIVL`, from the producer cache `round4/detach/cache/<shot>.npz`, else `detach_default/<shot>.npz` | MW (BOLOM labels the node MW but stores W; the row divides by 1e6) |
| Prad,div upper | `\BOLOM::PRAD_DIVU`, same | MW |
| Gas total (valve sum) | the corpus `gas_flow` valves GASA-E, LOB1-2, PFX1-3, UOB (`\D3D::TOP.NEUTRALS.GASFLOW.<valve>:FLOW`), each less its pre-plasma median, summed; a block where any listed valve has no finite mean is a gap, not a sum that counts it as zero. DIII-D has no calibrated total: the NEUTRALS tree has no total leaf and PTDATA `GASTOTAL` sums the raw valve volts | Torr L/s |
| Divertor Te (DTS) | PTDATA `tssdivte00`-`05`, the divertor Thomson Te the plasma control system receives, from `detach_default/dts/<shot>.npz`: one line per live channel, labelled with its blessed chord's Z | eV (PTDATA labels them "vo") |
| IRTV heat flux (W/cm²) | `\IRTVQ2DR2` = `\IRTV::TOP.IRTV:LODIV_60RP2:DIGITAL_CAM:HEAT_FLUX2D` (else `\IRTVQ2DR201`, the DIG_CAM_01 system), from `detach_default/irtv/<shot>.npz`: an R x time map of the lower divertor | W/cm² |

The bolometer chords, D-alpha chords, CO2 and Thomson density and the producer's f_div,
TangTV front and auxiliary rows left the default; `indicator_panels` still reads every
producer indicator, and the picker reaches the corpus groups. The camera frames and the
producer context (vote lanes, recipe, geometry) are unchanged.

### Afrac lower: coverage and the ungated line

At panel version 8, 7 of the 134 detachment stores showed any Afrac value: the producer
has a valid Afrac on 14 of its 470 shots, mostly lost to `short_reference`,
`probe_off_separatrix` and `probe_flux_unknown`. Since version 9 the invalid bins show the
ungated ratio as a second line (`ungated_afrac`), as `afrac.afrac_indicator` computes it:
each probe's inter-ELM bin median Jsat over `<ne>^2 P_SOL^(-3/7)`, over that probe's
reference, read on the probe nearest the separatrix. Removed: the power, ramp, ELM-share,
L-mode and ELM-coverage gates, the flux window on the read probe and the minimum reference
length. A probe's reference is the producer's own where one of its valid bins names it (so
the lines meet), else the 0.9 quantile of its ratio over its bins within |psi_N-1| <= 0.01,
else over all its bins. psi_N comes from the producer's flux map
(`round4/detach/efit/<shot>.npz`), else from `equilibria/<shot>.npz`; without a D-alpha ELM
mask no sample is dropped. The method readout gives each invalid bin's reason and how its
ungated value was read (`invalid (<reason>); ungated: P<n>, |psiN-1| ..., <reference>,
<flux map>`). An ungated bin never votes. A store without any value keeps an empty
"Afrac lower" row whose caveat ends "No value on this shot: <reason>" (no producer bins,
no Afrac in the plasma window, or why no ungated value could be read); without a plasma
window it spans the other rows. An invalid bin without an ungated value still reads
`invalid (<reason>); ungated: no value (<why>)`. At v9 (134 stores, 2026-10-17): valid
bins on 7 (all also with ungated bins), the ungated line only on 68, an empty row on 58
(not in the producer roster 42, no positioned probes 14, no CO2 line density 2), and no
row on 138388 (no rows at all: no BOLOM tree, no DTS, no IRTV).

### Divertor Te (DTS)

`tssdivteNN` is sample-and-hold at 0.25 ms, updated once a laser pulse (about 20 ms);
PTDATA labels it "vo" but it holds Te in eV, 0 where the fit failed (drawn as a gap).
Channel k follows blessed divertor chord k (`\ELECTRONS::TOP.TS.BLESSED.DIVERTOR:R/Z`):
over 371 channel-shots on 92 of the 134 stores, the nearest blessed chord in median
relative difference was chord k on 84 % (median 9 %); on the other 16 % no chord came
within 19 %. It is a real-time estimate, not the blessed fit. The row keeps only the update instants (a value changes; instants under
1 ms apart merge), steps from each to the next, and ends a step after two median periods.
All six chords sit at R = 1.485 m; their Z changes with the Thomson configuration (all
lower, all upper, or split, e.g. 0-1 upper and 2-5 lower from 189057), so the legend gives
each line's Z. The pointnames do not exist before about 176000 (PTDATA answers
"getservbyname failed for task 'PTSERVER'" there, while other pointnames read).
Over the 134 stores (2026-10-16): the pointnames read on 123 (not on the 10 shots before
176011, nor on 200007), hold live values on 94 and only zeros on 29. The blessed chords
read on 105 (8 to 82 chords; tssdivte follows the first six). Chord Z sets of the 100
stores with both: 0-1 upper and 2-5 lower on 72, all lower on 17, 0-3 upper and 4-5
lower on 5 (186xxx), all upper on 6. At v9 the row is on the 94 live stores (6 live
channels on 46, 5 on 34, 4 on 12, 3 on 2).

### IRTV heat flux

HEAT_FLUX2D is the IRTV analysis's surface heat flux, (t, pixel) float32 with `dim_of(0)`
= R (cm), `dim_of(1)` = t (ms) and `dim_of(2)` = Z (cm) per pixel, units "W/cm^2", at the
camera rate (12 kHz on 201072, 0.083 ms on 190515; 7.9 ms on 191506). The pixels run
down the centre post (R 104.7 cm, Z -79 to -119 cm), along the floor (Z -136 cm) and onto
the shelf (Z -124.5 cm, R to 166 cm); R never decreases. The fetch keeps 1 ms means of
each pixel (float16, clipped at 65,000), on blocks aligned to whole milliseconds; the row
places each block at its centre (so the map never refines the ~1 ms store grid), averages
the pixels in 0.5 cm R bins and
leaves out a bin whose pixels span more than 2 cm in Z (the centre post, one radius at
many heights; `metadata.wall_left_out`). Colour runs from 0 to the window's 99.5th
percentile. Matches the owner's `irtv_12_03_2024.pkl` (shots 201072-201085) to 0.01 cm in
position and a row correlation of 0.998-0.9999. The temperature twin `\IRTVAT2DR2`
(`ABS_T_2D`, deg C) exists wherever HEAT_FLUX2D does; it is not a row. Over the 134
stores: HEAT_FLUX2D (DIGITAL_CAM) on 95, from 158019 on (DIG_CAM_01 on 87, DIG_CAM_02 on
26, never alone); the 165- and 225-degree views on none; PERI75R0 on 88.
At v9 the row is on those 95 stores, all from `\IRTVQ2DR2` (DIGITAL_CAM).
On the four stores without a plasma window the grid spans every row, so the map moved it:
158019 from 4 ms to 1 ms steps (11418 bins); 170816, 178633 and 178636 start 0.15 ms
earlier (2856 bins).

## Slices (`src/labeler/events/review/slices.py`, `ui/slice_routes.py`)

- `GET /api/slices?event&shot` -> `{kinds: [{kind, label, n_t, t_range_ms}]}`: what the shot
  has on disk.
- `GET /api/slice?event&shot&kind=equilibrium&t_ms` -> JSON `{t_ms, source, source_rule,
  r_range, z_range, boundary, xpoints, strikes, wall, levels}` in metres: the EFIT slice
  nearest `t_ms`, its separatrix (RBBBS/ZBBBS), real X-points and strike points (the EFIT
  sentinels -9.99, -0.89 and 0 dropped), the wall (LIM) and psi_N contours at 0.2, 0.4,
  0.6, 0.8, 1.0, 1.02 and 1.05, cut where they leave the wall.
- `GET /api/slice?event&shot&kind=inversion&t_ms` -> grayscale PNG of the nearest
  inversion frame, row 0 = z_max and column 0 = r_min, scaled between the shot's 1st and
  99.5th percentiles; header `X-Slice` = `{t_ms, r_range, z_range, vmin, vmax, units,
  source}` (units "a.u.": the inversions record none).
- 404 when the shot has nothing of that kind, no slice within 40 ms, or an unreadable
  file (the message names it); 422 for a non-integer shot or `t_ms`, or an unknown kind
  or event. Any event may ask.
- The listing reads only each file's clock (cached by file, mtime and size) and leaves
  out a file it cannot read. Inversion radii stored decreasing are flipped. An
  equilibrium without a psi grid or a wall takes its axes from the boundary.

Loaded files are cached by path and modification time, slices and PNGs by file and index.

## Caches (`$LABELER_ROOT`)

- `equilibria/<shot>.npz`: EFIT02, or EFIT01 when EFIT02 is missing or has under two
  slices (`source` says which): GEQDSK RBBBS/ZBBBS/NBBBS/LIM, AEQDSK X-points and strike
  points on the GEQDSK clock (0.5 ms match), PSIRZ (t, z, r) with R, Z, SSIMAG, SSIBRY
  when the grid is at most 65 x 65; float32, compressed. 192 shots, 563 MB (2026-10-16):
  141 EFIT02, 17 EFIT01 (EFIT02 sparse), 34 EFIT01 (no EFIT02). 190271 keeps its boundary,
  X-points and wall but no psi grid: its square PSIRZ matches neither axis order.
- `detach_default/<shot>.npz`: Prad,div for shots the producer cache lacks (41 shots, 1.9 MB;
  138388 has no BOLOM tree and 190046 aborts in TDI, so neither has Prad rows).
- `detach_default/dts/<shot>.npz`: `tssdivte00`-`05` (`<name>__t`, `<name>__y`, float32)
  and the blessed divertor chords (`chord_r_m`, `chord_z_m`, `chord_phi_m`), with
  `status` (JSON: each pointname's units and live count, or its error). 123 files, 24.7 MB
  (none for the 11 shots without the pointnames).
- `detach_default/irtv/<shot>.npz`: `heat_flux` (1 ms blocks x pixels, float16), `t_ms`,
  `r_cm`, `z_cm`, `node`, `system`, `units`. 95 files, 279 MB (at most 5.8 MB a shot);
  the fetch log names what was tried on the 39 shots without a map.
- `round4/detach/inversions/<shot>.npz`: the producer's TangTV inversions (read only).

`scripts/labeler/detach_default_fetch.py` writes them on the login node (flock + `fdp
run`, paced; `--parts equilibrium prad dts irtv`, default all). It resumes by part: the
log `runs/detach_default/fetch_log.jsonl` records `done_parts` per shot (older entries
count as equilibrium + prad), and it exits 3 on an authentication error (a whole word
such as token, login, 401 or one starting auth/unauth, not a substring). The view itself
never fetches.
