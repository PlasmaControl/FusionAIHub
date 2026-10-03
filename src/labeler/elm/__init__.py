"""ELM detection on the reviewed spans: `elm-ours`, the lab's DSM as a detector, the reference swap.

`inputs` brings the fetched filterscope and interferometer records to a 10 kHz
grid, `labels` turns the owner's reviewed spans into dense targets and scored 50 ms
bins, `net` is the 1-D U-Net, `train` fits it in shot-grouped cross-validation,
`score` scores any per-bin output against the bins with shot-bootstrap intervals,
`dsm` re-reads the lab's ELM survival model as a detector, and `swap` converts the
legacy onset table to the same bins for the reference swap. The write-up is
`docs/labeler/elm_ours.md`.
"""
