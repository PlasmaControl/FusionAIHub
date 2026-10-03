"""Whole-interval tearing-mode labels, their agreement with the archives, and a detector.

`rule` turns a shot's n = 1 and n = 2 magnetic RMS (`\\MHD::N1RMS`, `\\MHD::N2RMS`)
into the intervals over which a tearing mode is present. `agreement` compares those
with the lab's onset labels (Seo's archive, the survival labels). `scoring` holds the
per-bin scores and the shot bootstrap every detector in the benchmark is judged by.
Nothing here imports torch, h5py or MDSplus.
"""
