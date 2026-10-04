"""Resistive wall mode baselines: labels, features and a balanced forest.

`rwm-brf` adapts Piccione et al. 2022 (NSTX) to DIII-D; `rwm-nnpu` trains the
same features with a non-negative positive-unlabelled risk (Kiryo et al. 2017),
because the only RWM labels are onsets from Jeremy Hanson's database and an
unlisted shot is unlabelled, not negative.
"""
