"""MHD mode numbers from the magnetics, ECE, CO2, EFIT02 and the plasma rotation.

The dev/mhd method (2026-10-09 to 10-13), ported as importable modules:

- `tracks`: Mirnov peaks every 4 ms, the toroidal n fit, frame-to-frame tracks (n = 1-3),
  and per frame the wall parity, the harmonic-lock sub-windows, ECE and EFIT02.
- `rules`: the gates, the harmonic lock, the track table and rules 1-4 (fixed-frequency
  lines, harmonics, the q-surface and parity candidates).
- `views`: every channel's cross-spectrum with the magnetics, summed over a track, and the
  ECE locators.
- `labelmodel`: the unified label model (one latent m per track, every view one shared law).
- `resonance`: the rotating-resonance gate and the track's status (m/n, q_min above,
  uncertain, not resonant, no rotation).
- `freeze`: the one-off fit on the dev corpus (`freeze.json`).
- `pipeline`: one shot from the corpus record to its tracks, cached per shot.
- `rows`: the eight MHD events' suggestion rows from a shot's tracks.
"""
