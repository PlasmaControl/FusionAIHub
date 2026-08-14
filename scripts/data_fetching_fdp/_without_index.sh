#!/bin/bash
# Drop the PTDATA federation index from the environment, then exec.
# With the index absent, PTDATA-backed TDI expression nodes read the staged
# objects (via SYS_D3) instead of the federation: ECE TECEF01 goes from 18.9 s
# to 1.34 s. Used by fetch_shot_fast.sh for the MDSplus pass only.
unset PTDATA_JSON_INDEX_DIR
exec "$@"
