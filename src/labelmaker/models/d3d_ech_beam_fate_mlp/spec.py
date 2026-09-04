"""d3d_ech_beam_fate_mlp - scaffold. See README.md for what this model is and what blocks it."""

raise NotImplementedError(
    "d3d_ech_beam_fate_mlp is a scaffold: no adapter yet. "
    "Upstream weights: /projects/EKOLEMEN/ECH_interlock/models_v15/. "
    "Blocked on: the file is a legacy Keras-2 HDF5 containing `TFOpLambda`, which `runners/keras_h5.py` does not implement - add that layer or re-save the graph once via `tf_keras`; its profiles are on a 101-point grid, so `ne` and `Te` need their own canonical features rather than the 33-point `RHO_GRID`; confirm whether the normalisation constants in `norm_stats.npy` are already baked into the graph"
)
