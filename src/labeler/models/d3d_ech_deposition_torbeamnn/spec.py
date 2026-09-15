"""d3d_ech_deposition_torbeamnn - scaffold. See README.md for what this model is and what blocks it."""

raise NotImplementedError(
    "d3d_ech_deposition_torbeamnn is a scaffold: no adapter yet. "
    "Upstream weights: /projects/EKOLEMEN/torbeamNN/models_v2/. "
    "Blocked on: recover the input list and ordering from the torbeamNN training code; O-mode and X-mode are separate graphs, so the adapter needs the per-gyrotron polarisation, which the corpus does carry as `ech_polarization`; decide whether to emit one label per gyrotron or one aggregate"
)
