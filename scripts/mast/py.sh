#!/usr/bin/env bash
export LD_PRELOAD=/usr/local/cuda-13.5/lib64/libnvrtc.so.13:/usr/local/cuda-13.5/lib64/libnvrtc-builtins.so.13.5${LD_PRELOAD:+:$LD_PRELOAD}
exec "${IGNITE_MAST_ENV:-/models/envs/ignite-mast}/bin/python" -I "$@"
