# IGNITE on MAST, stage 1 ("Tokenizers"): index of the three implementation plans

> Approved by the owner on 2026-11-22 (16:17 UTC, "yes i approve the plan. please carry it out"). The open decisions for the owner are taken as the author recommends until the owner says otherwise. The stage-0 numbers quoted in this text are those of the day it was written; `docs/models/ignite-mast-stage0.md` holds the current ones. Written 2026-11-22 by the plans' author. Nothing in the three plans has been implemented or run; the only measurements in them are stage 0's and the author's read-only looks at cache v1, each marked as such.

**Spec:** `docs/models/ignite-mast-design.md`, row "1. Tokenizers" of "Order of work": "the fixed-transform slow tokenizer with K per signal; the learned profile and map tokenizers against their linear baselines; the fluctuation path; the learned slow tokenizer and the two controls as comparisons; the headroom table for the other seven tasks".

Stage 1 is three subsystems that produce working, testable software on their own, so it is three plans. Each is complete: its own open decisions for the owner, constraints, rules stated before the numbers, tasks, controller jobs.

| Order | Plan | What it delivers | Tasks | Needs |
|---|---|---|---|---|
| 1, at once | `2026-11-22-ignite-mast-stage1a-headroom-seven.md` | The headroom table with a measured row for each of the 14 tasks; a direct model per task as an instrument for 1b and 1c | 4 | Stage 0's code only. Its trainings (job J4: 11 runs, one wave of 5 to 7 h on the four GPUs) are the longest jobs of stage 1 |
| 1, beside 1a | `2026-11-22-ignite-mast-stage1b-fixed-tokenizers.md` | Tokenizers without learned weights for all 39 signals (cosine transform per frame, principal components for profiles and the flux map, band transform for the Mirnov spectrograms), K per signal by a ladder, the gate, the representation floor of our tokenizer per task through the stage-0 harness, and the assignment `v1` that stage 2 starts on | 11 | Stage 0's code. From 1a only the direct models of tasks 1-3 and 3-2, and only for signals that miss their target |
| 2 | `2026-11-22-ignite-mast-stage1c-learned-tokenizers.md` | Learned tokenizers shared per family, the specialized controls, the three-arm comparison (shared, own diagnostic, concatenated) in linear form and as networks, and the assignments `v2` and `v2c` | 8 | 1b complete (its Tasks 1 to 10 and jobs K1 to K3) |

Dates: stage 1's target is 2026-11-27; the node is available until about 2026-12-12. The full FAIR-MAST data (expected about 2026-11-25) is not a dependency of any of the three plans. Stage 2 does not wait for 1c: it starts on 1b's `v1`, and `v2` replaces it only where 1c's rules say so.

Where stage 0 stands (2026-11-22, `docs/models/ignite-mast-stage0.md`): the projected 14-task mean is 0.1999 against the bar of 0.183 (0.194 with the best published score everywhere). By the design's text the headroom check did not pass with the direct model, and the bar is with the owner. No plan here assumes a changed bar or a passed check. Stage 1 is not gated by the check; stage 2 is.

## What 1c needs 1b to keep

1c is written against 1b's plan, not against code. If the review of 1b changes one of these, the matching lines of 1c change with it:

- `assign.Setting` has the field `source: str = ""`, part of the hashed JSON of an assignment.
- `tokenizers.code_sha256()` hashes the `*.py` files directly in `ignite/mast/tokenizers/` and does not descend into `tokenizers/learned/`, so 1c's code does not make 1b's stored scores stale.
- `ladder.values_per_frame` counts the two codec names `recoded` and `recoded_concat`, so 1b's gate works on an assignment of 1c without change.
- `ladder.shot_error`, `ladder.band_contrast` and the gate's cache side decide "spectrogram" by a codec's `domain == "band"`, not by its class.
- `fit.build_codec` raises `ValueError` for a codec name it does not know; 1c's builder handles its own two names first.
- The `Codec` base class (`encode_domain`, `decode_domain`, `domain_from_model`, `roundtrip`), `codecs.fit_pca`, `grid.GridSet`, `grid.fit_ranges`, and in `scripts/mast/run_tokenizer.py` the functions `provenance_of`, `kept_score`, `predictor_name` and the sub-commands `fit`, `score`, `ladder`, `assign`, `gate`.

## Open decisions for the owner, across the three plans

Each plan lists its own at its top, with the author's recommendation. In one place:

- 1a: (1) what the 14-task projection does to the bar; (2) whether a task without headroom keeps its place in the 14-task mean.
- 1b: (1) task 4-5 and frame statistics; (2) the linear baseline may stand as the tokenizer for profiles and the flux map; (3) what is published about signals that miss the fidelity target.
- 1c: (1) what a win of the concatenated control means; (2) token count as a criterion; (3) channel identity without positions; (4) no fluctuation tokens for the 50 kHz channels in stage 1; (5) the cut order if stage 1 is late.

## Size of the plans

1a is 35 kB, 1b 141 kB, 1c 128 kB; the design document is 59 kB. No step in them carries a function body. In 1b, 41% of the bytes are test code and commands (test names and assertions), 26% the Interfaces blocks (signatures with one sentence each), 16% the argument at the top (open decisions, measured facts, departures from the design, the rules), and the rest run commands, expected outputs, trials and controller jobs. 1c has the same make-up. They are long because the surface is wide (1b: 11 tasks, 73 tests, 5 sub-commands; 1c: 8 tasks, 44 tests, 9 sub-commands), not because code was written out. Shortening them further means cutting scope, which is a decision for the review, not an edit.
