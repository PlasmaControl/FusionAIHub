# Detachment default view (plan D.B)

The owner's 2026-10-15 request: the detachment default needs Afrac lower and upper,
Prad,div lower and upper, the gas total, the IRTV temperature, and 2-D views of EFIT and
the TangTV inversion; not the individual bolometer and D-alpha chords. IRTV holds no
temperature trace, so the IRTV rows are two maps: the heat flux and, since version 10
(owner, 2026-10-17), the absolute surface temperature right after it; Afrac upper has no
source. Also since version 10 the divertor Te row is the blessed Thomson fit as a Z x time
map; the real-time `tssdivte` lines left the default.

## Rows (`src/labeler/events/panels/detachment.py`, store `PANEL_VERSIONS` 10)

In this order; a row whose source is not verified is left out, never guessed.

| Row | Source | Units |
|---|---|---|
| Afrac lower | the labeler's Afrac bins, `round4/detach/bins/<shot>.npz` (`indicator_path`): an uncalibrated local Jsat ratio at the lower outer strike point, not the published Eldon Afrac (caveat on the row). Line 1: the producer's valid bins. Line 2 ("ungated, invalid bins"): on the bins the producer marks invalid, the same ratio with its gates removed (below) | dimensionless, step bins |
| Afrac upper | **left out (owner, 2026-10-16)**: no node, PTDATA pointname or LANGMUIR leaf found | |
| Prad,div lower | `\BOLOM::PRAD_DIVL`, from the producer cache `round4/detach/cache/<shot>.npz`, else `detach_default/<shot>.npz` | MW (BOLOM labels the node MW but stores W; the row divides by 1e6) |
| Prad,div upper | `\BOLOM::PRAD_DIVU`, same | MW |
| Gas total (valve sum) | the corpus `gas_flow` valves GASA-E, LOB1-2, PFX1-3, UOB (`\D3D::TOP.NEUTRALS.GASFLOW.<valve>:FLOW`), each less its pre-plasma median, summed; a block where any listed valve has no finite mean is a gap, not a sum that counts it as zero. DIII-D has no calibrated total: the NEUTRALS tree has no total leaf and PTDATA `GASTOTAL` sums the raw valve volts | Torr L/s |
| Divertor Te (blessed) | `\ELECTRONS::TOP.TS.BLESSED.DIVERTOR:TEMP` (owner, 2026-10-17, replacing the real-time `tssdivte00`-`05` lines), from `detach_default/dts_blessed/<shot>.npz`: a Z x time map, one profile per laser pulse, log colour 0.5-100 eV | eV |
| IRTV heat flux (W/cm²) | `\IRTVQ2DR2` = `\IRTV::TOP.IRTV:LODIV_60RP2:DIGITAL_CAM:HEAT_FLUX2D` (else `\IRTVQ2DR201`, the DIG_CAM_01 system), from `detach_default/irtv/<shot>.npz`: an R x time map of the lower divertor | W/cm² |
| IRTV temperature (°C) | `\IRTVAT2DR2` = `\IRTV::TOP.IRTV:LODIV_60RP2:DIGITAL_CAM:ABS_T_2D` (else `\IRTVAT2DR201`), from `detach_default/irtv_temp/<shot>.npz`: the heat-flux map's twin on the same clock and pixels, same 1 ms means and 0.5 cm R bins | deg C |

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
row on 138388 (no rows at all: no BOLOM tree, no DTS, no IRTV). The Afrac row is
unchanged at v10 (owner, 2026-10-17); 138388 now has an empty Afrac row beside the
blessed Te row, which it does have.

### Divertor Te (blessed)

`\ELECTRONS::TOP.TS.BLESSED.DIVERTOR:TEMP` is the Thomson fit, (chord, time) float32 in
eV, 0 where the fit failed, one profile per laser pulse (about 20 ms), with the chord
positions `:R` (always 1.485 m) and `:Z` and `:PHI`. The fetch (`dts_blessed`) keeps
`temp`, `t_ms`, `r_m`, `z_m`, `phi_deg` and `units`; an empty node or fewer than two
profiles gives no file. The row (`dts_panel`) is a map: Z (m) on the vertical axis, time
across, Te as log10 colour pinned to 0.5-100 eV (black is no measurement or below 0.5 eV,
the top colour is 100 eV or more; fits above 100 eV occur and can be spurious). The pin
keeps one colour scale across shots; `metadata.z_scale = "log"`, which `panel_rows`
reads (its `_log` helper scales log10(Te) linearly into bytes between the pins and
reports `z_lo`/`z_hi` back in eV: 0.5 and 100). The vertical axis is uniform 0.5 cm bins of Z
(`DTS_Z_BIN_M`); each live chord fills the bins within 1.5 cm of it (`DTS_REACH_M`,
nearest chord wins), and wider gaps stay empty. A dead chord (all zeros in the window)
is dropped; a window with fewer than two profiles has no row. Without a plasma window
the row spans the whole file, and so may the grid of the store (below). The row is
nearest-sampled in time, so each 20 ms profile is painted about 10 ms either side of its
time.

The chords move with the Thomson configuration, so the vertical axis would hold a lot of
empty space where a minor array sits (typically 14 chords from Z -1.24 to -1.01 m plus 1
or 2 near Z +1 m). The display band hides a minor array: the chords split into arrays at
gaps above 0.3 m, an array with fewer than 5 chords (`DTS_MINOR_CHORDS`) is minor, and
the row's `band` (the visible bins the page honours) spans the other arrays; the minor
chords stay in the row and are named in the caveat and in `metadata.band_z_m`,
`chords_outside_band_z_m`, `chord_z_m` and `chord_r_m`. A shot with one array, or with
only big arrays, hides nothing; when every array is minor the longest one is shown.

Over the 134 stores (2026-10-17): the node reads on 104 (30 have none: 29 TreeNODATA, 1
with a single profile); 8 to 82 chords per store (8 on 4, 14 on 13, 16 on 83, 82 on 4),
mostly 14 lower + 2 upper (about 64 stores), 14 lower + 1 upper (8), 14 lower only (8),
12 lower + 4 upper (5), 4 lower + 12 upper (5). At v10 the row is on the 104 stores
(45 to 502 Z bins); the chords it draws (dead ones dropped) are 16 on 70, 14 on 15, 15 on
11, 8 on 4 and 9, 13, 38 and 81 on one each. The band hides a minor array on 83 stores (1
chord on 9, 2 on 64, 4 on 10) and everything is shown on 21. The real-time
`tssdivte00`-`05` pointnames (sample-and-hold, labelled "vo", 0.25 ms) left the default:
the `dts` fetch part stays selectable (`--parts dts`) and its files in `detach_default/dts/`
are kept, but nothing reads them. They followed the blessed chord k in median relative
difference on 84 % of 371 channel-shots over 92 stores, and the pointnames exist only
from about 176000.

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
position and a row correlation of 0.998-0.9999. Over the 134
stores: HEAT_FLUX2D (DIGITAL_CAM) on 95, from 158019 on (DIG_CAM_01 on 87, DIG_CAM_02 on
26, never alone); the 165- and 225-degree views on none; PERI75R0 on 88.
At v9 the row is on those 95 stores, all from `\IRTVQ2DR2` (DIGITAL_CAM).
On the four stores without a plasma window the grid spans every row, so the map moved it:
158019 from 4 ms to 1 ms steps (11418 bins); 170816, 178633 and 178636 start 0.15 ms
earlier (2856 bins).

### IRTV temperature

Owner, 2026-10-17: the absolute surface temperature as a second IRTV row, right after the
heat flux. `\IRTVAT2DR2` is `ABS_T_2D` (deg C) of the same analysis, with the same shape,
clock and pixel positions as HEAT_FLUX2D (checked on all 95 files), so the row is built
the same way: `irtv_panel(..., quantity="temperature")` reads `detach_default/irtv_temp/
<shot>.npz` (`temperature`, `t_ms`, `r_cm`, `z_cm`, `node`, `system`, `units`), takes 1 ms
means (float16 like the heat flux), places each block at its centre, averages the pixels
in 0.5 cm R bins and leaves out a bin whose pixels span more than 2 cm in Z. The colour
is linear between the window's 1st and 99.5th percentile (the heat flux pins its floor
at 0; a temperature has no natural zero, so neither end is pinned): over the 95 stores
the colour range `z_lo`-`z_hi` has medians 26 and 240 deg C (low end 0.8 to 72, high end
65 to 605). The fetch (`irtv_temp` part) found `\IRTVAT2DR2` on 95 shots (the DIG_CAM_01
fallback `\IRTVAT2DR201` on none), 218 MB (at most 4.6 MB a shot); the 39 other shots
have an empty node and no row, and no store has only one of the two IRTV maps.

On the stores without a plasma window the grid spans every row, so the blessed Te row
widened it where the DTS record runs past the other rows (v10): 138388 (now with the Afrac
and Te rows, 8 chords), 156785 (2855 to 3020 bins), seven 189xxx stores (11601 to
12878-13000 bins: 189329, 189378, 189390, 189597, 189598, 189602, 189603), 190670 and
192711 (11721 to 13060 and 11730 to 13129). All 107 stores with a plasma window and 16 of
the 27 without one keep the v9 grid, and the camera frames of all 134 stores are identical
to v9 (22,936 frames in 171 channels). On the 123 stores with an unchanged grid every
other row is identical to v9 (561 rows compared, none differs).

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
- `detach_default/dts_blessed/<shot>.npz`: the blessed divertor Te, `temp` (chord, time,
  float32 eV, 0 = no fit), `t_ms`, `r_m`, `z_m`, `phi_deg`, `units`. 104 files, 1.2 MB (the
  30 other shots have no data on the node or a single profile).
- `detach_default/irtv_temp/<shot>.npz`: `temperature` (1 ms blocks x pixels, float16),
  `t_ms`, `r_cm`, `z_cm`, `node`, `system`, `units`. 95 files, 218 MB.
- `detach_default/dts/<shot>.npz` (retired from the default at v10, kept): `tssdivte00`-`05`
  (`<name>__t`, `<name>__y`, float32) and the blessed divertor chords (`chord_r_m`,
  `chord_z_m`, `chord_phi_m`), with `status` (JSON: each pointname's units and live count,
  or its error). 123 files, 24.7 MB (none for the 11 shots without the pointnames). Nothing
  reads them any more.
- `detach_default/irtv/<shot>.npz`: `heat_flux` (1 ms blocks x pixels, float16), `t_ms`,
  `r_cm`, `z_cm`, `node`, `system`, `units`. 95 files, 279 MB (at most 5.8 MB a shot);
  the fetch log names what was tried on the 39 shots without a map.
- `round4/detach/inversions/<shot>.npz`: the producer's TangTV inversions (read only).

`scripts/labeler/detach_default_fetch.py` writes them on the login node (flock + `fdp
run`, paced; `--parts equilibrium prad dts irtv dts_blessed irtv_temp`, default all but
the retired `dts`). It resumes by part: the log `runs/detach_default/fetch_log.jsonl` records `done_parts` per shot (older entries
count as equilibrium + prad), and it exits 3 on an authentication error (a whole word
such as token, login, 401 or one starting auth/unauth, not a substring). The view itself
never fetches.
