"""Alfven-eigenmode label primitives (spec section 5).

`labels` holds the pure numpy rules that turn tokeye's two-channel coherent /
transient mask into the frame-level AE activity label. Nothing here imports
torch, tokeye or h5py, so the same code runs in the pixi test env, in the
tokeye venv that owns the GPU, and later inside the labelmaker adapter.
"""
