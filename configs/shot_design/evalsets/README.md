# `ideate` evaluation sets

Two frozen files and a rule:

| file | what it is |
| --- | --- |
| `reference_shot_prompts.csv` | 200 prompts a DIII-D physicist would type into a shot recommender |
| `split.yaml` | the dev / eval split of the 500 `recommender_v1` shots, cut by run day |

Both are **frozen**. `tests/ideate/test_evalset_frozen.py` asserts the CSV's sha256 against a
literal in the test *and* against the hash recorded below, and re-derives `split.yaml` from the
rule over the committed shot list. Editing the evalset therefore takes three deliberate edits in
three files and shows up in every diff.

```
version: 1.1
sha256(reference_shot_prompts.csv) = aed8547330414e5b427579b58b207d661609bbfca1b048ca6df666e71172ef0e
superseded: v1.0 = a6059fbccd3aec79547d5a7ced3896c0ea6496c4e60927aa108675caef63a6a9
```

The superseded hash is kept so that a number published against v1.0 can still be told apart from
a v1.1 one. **Always quote the version beside the number.**

## Changelog

### v1.1 — 2026-09-13, the one re-freeze, before any retrieval tuning

An independent review of v1.0 found eleven prompts with defects in the **question** and ten rows
whose emptiness needed recording. These are corrections to what is asked, not to what was
answered: nothing in retrieval, the ranking, the scoring or the lexicon-matching code changed
alongside them, and the re-run is reported against both hashes. The `fast_ion` phenomenon id that
four of these rows now expect was added at the same time — to `src/labelmaker/events/lexicons.yaml`
and `configs/ideate/phenomena.yaml` as a **text-only topic with no detector and no label**,
deliberately *not* as four new aliases on `ae` (see the note in either file).

**Eleven prompts changed:**

| prompt_id | v1.0 | v1.1 | why |
| --- | --- | --- | --- |
| `p036` | expected `ae` | expects `fast_ion` | a 3D-field fast-ion transport experiment names no mode; `ae` is the AE *mode* id, not a fast-ion topic |
| `p038` | expected `ae` | expects `fast_ion` | FIDA is a diagnostic; the losses may be from 3D fields, ripple or sawteeth |
| `p041` | expected `ae` | expects `fast_ion` | the sentence names no mode at all |
| `p046` | expected `ae` | expects `fast_ion` | a neutron deficit is a fast-ion transport signature, not an AE claim |
| `p066` | expected `sawtooth\|ae` | expects `sawtooth\|fast_ion` | Porcelli fast-ion stabilisation of the internal kink — no Alfvén eigenmode in it |
| `p197` | expected `ae\|sawtooth` | expects `sawtooth\|fast_ion` | "fast ion driven modes" is the topic, not a named AE |
| `p048` | expected `ae\|fishbone` | expects `fishbone` | below 40 kHz contradicts the registry's own AE band, so requiring both was unresolvable by construction |
| `p039` | "TAE-induced ITG turbulence" | "interaction between TAEs and ITG turbulence" | garbled physics — TAEs do not induce ITG turbulence. Still a deliberate plural-gap probe (`tae` is an alias, `taes` is not) |
| `p072` | "sawtooth behaviour in a hybrid with qmin above 1", expected `sawtooth` | "sawtooth-free hybrid operation with qmin above 1", expects nothing | self-contradictory: qmin > 1 means no q = 1 surface and so no sawteeth, and the hard filter selected *against* the expected phenomenon |
| `p186` | "L-H transition with ECH only and no beams" | "L-H transition at the ITER-similar shape with the beams held flat", `kappa_mean ≥ 1.7` | byte-identical expectation to `p079`, so one unanswerable question was charged twice against coverage |
| `p053` | — | unchanged, annotated | it misses only on the plural: the lexicon has `ntm`, not `ntms`, where `elm` has both |

**What this did to the `fast_ions` number, stated so nobody quotes it wrongly.** The category's
resolution went from **56.2 % FAIL (v1.0)** to **81.2 % PASS (v1.1)**, and *both* halves of the
test moved to produce that: the four rows above stopped expecting `ae`, **and** the aliases they
now resolve through (`fast ion(s)`, `beam ion(s)`, `energetic ion(s)`, `fida`) were harvested from
those same four sentences. Before the id existed all four resolved to nothing; after it, all four
resolve to the id made for them. Under the review's alternative remedy — leave those rows with an
empty `expect_phenomena` and change no vocabulary — the category reads **9/12 = 75.0 %, still
FAIL**. So `fast_ions 81.2 %` is not an independent test of the lexicon against unseen wording and
is not comparable to the 56.2 %; the honest next measurement is an *unseen* set of fast-ion
sentences in a future v2.

**Ten rows annotated, not changed.** Eight are **unanswerable on this corpus by construction**
(`p079`, `p100`, `p111`, `p120`, `p125`, `p134`, `p163`, `p170`) and three are satisfiable
corpus-wide but empty on the eval side (`p028` 8 shots, `p123` 6, `p166` 8). Measured
2026-09-13 over all 500 `recommender_v1` flat-tops:

| ceiling | measured |
| --- | --- |
| `betan_mean` | 0.072 – **3.006** — `betaN ≥ 4` matches nothing |
| `pech_total_mean` | 82 W – **2.298 MW** — `≥ 3 MW` matches nothing |
| `pnbi_total_mean` | **0.834 MW** – 11.9 MW — neither `≤ 0.5 MW` nor `≥ 12 MW` matches anything |
| `q95_mean` | **NaN on all 500 shots** — the builder writes the column, nothing records it |

They are kept, not deleted. They are honest things for a physicist to type, "nothing here" is the
correct answer, and the annotation is what stops the next reader re-diagnosing them as retrieval
misses. But they do cost coverage, and that is why coverage alone is not the whole story.

## Why frozen

An evaluation set that can be edited after the numbers come in measures nothing. The temptation
is never to cheat outright; it is to "fix a badly worded prompt" that happens to be the one that
missed, and after the fact nobody — including the person who did it — can tell that apart from a
genuine correction. So the set was authored first, committed, hashed, and only then run. **A miss
is a finding.** The right response to a prompt the retrieval cannot answer is a line in the
report, not a rewrite of the prompt.

## How the 200 prompts were authored

They are written in the vocabulary this corpus actually uses, taken from four sources, all
read-only:

1. **The mini-proposal titles of the 500 `recommender_v1` shots** (`shots.parquet.mp_title`) —
   "Feedback Adaptive RMP ELM Controller", "Isotope Effect in Standard and Wide Pedestal QH-Mode",
   "High power, maximum q‖ detachment", "Polarization Investigation of top launch ECCD",
   "Transport of passing and trapped fast ions by static 3D fields". These are what physicists
   call their own experiments.
2. **The operators' own logbook entries** (`HumanTier.log_entries`) — "Lost the 330s this time,
   and 2 of 4 gyros", "got tearing modes, n=2 is first", "Great shot with EHOs", "ELM suppression
   w RMP + Li", "3-5 kA n=3 RMP in even parity during 2.5-6s", "cut early gas (0 < t < 200ms)".
   Several prompts are these sentences nearly verbatim, because that is how the question actually
   arrives.
3. **The 12 phenomena of `configs/ideate/phenomena.yaml`** and the alias lists in
   `src/labelmaker/events/lexicons.yaml`, for `expect_phenomena`.
4. **The 14 curation themes of `configs/ideate/labels.yaml`** (`shotdb.select.lexicon_themes` /
   `assign_theme`), which is the vocabulary `ideate corpus select` filed the 500 shots under.
   There is no `themes.yaml`.

Prompts were **not** written by checking what `retrieval.phenomena.resolve` returns.
`expect_phenomena` records what a physicist typing that sentence *means*; the gap between that and
what the lexicon resolves is exactly what the harness measures. That gap is designed in: about a
third of the prompts use the operator shorthand that is *not* an alias — "wide pedestal QH",
"radiative divertor", "X-point radiator", "H-mode access", "island growth" — because a real user
types those and an evalset made only of alias-hits would measure nothing.

### Columns

| column | meaning |
| --- | --- |
| `prompt_id` | `p001`…`p200`, in file order |
| `category` | one of the 15 below |
| `prompt` | the text, exactly as it is fed to `rank.search` |
| `expect_phenomena` | `|`-separated registry ids, or empty — what the physicist means |
| `expect_constraints` | compact JSON `{"col":[lo,hi]}`, the same shape `--where COL=LO:HI` parses to |
| `expect_segment` | `flat_top` (most), `ramp_up`, `ramp_down` |
| `hand_graded` | `1` for exactly 20 rows |
| `notes` | the human rubric, on hand-graded rows only |

### Categories

| category | n | | category | n |
| --- | --- | --- | --- | --- |
| `qh_mode` | 16 | | `nbi_program` | 12 |
| `elm_rmp` | 16 | | `ech_program` | 12 |
| `fast_ions` | 16 | | `gas_puff` | 10 |
| `tearing` | 14 | | `reference_shot` | 12 |
| `detachment` | 14 | | `actuator_only` | 12 |
| `lh_transition` | 12 | | `negation` | 12 |
| `high_qmin` | 12 | | `mixed` | 20 |
| `sawtooth` | 10 | | | |

`qh_mode`, `elm_rmp` and `fast_ions` are named by the plan, which sets the ≥ 80 % phenomenon
resolution bar on those three. `negation` ("without ELMs", "no NBI", "attached divertor
throughout, no detachment") is there because absence is a claim the system is allowed to get
wrong in a specific dangerous way: a shot nobody looked at is not a negative.

### The 20 hand-graded prompts

Exactly 20 rows carry `hand_graded=1` and a rubric in `notes` saying what a correct top-5 looks
like — not a list of shot numbers, which would be a second evalset nobody had verified, but the
property the five have to have ("every shot in the top-5 must satisfy both ranges", "a forecast-only
hit must be labelled as a forecast and must not outrank an observed one", "no detector writes
detachment, so every hit here is text-only by construction and must say so"). They are spread over
at least 12 categories, at most 2 per category.

`src/ideate/eval/prompts.py` computes a **proxy** for these — does any of the top 5 carry evidence
of every expected phenomenon, or do all 5 satisfy the expected constraints — and every report
carries `ProxyGrade.caveat` saying, in the report itself, that it is not the human grade. A prompt
can pass the proxy with five useless shots and fail it while returning the five a physicist would
have picked.

## The split

```
eval iff int(sha256(run_id)[:8], 16) % 1000 < 200      # a run day, never a shot
```

Measured on `recommender_v1`: **dev 390 shots / 182 run days, eval 110 shots / 50 run days**, no
run day on both sides. 20 % is a round fraction of the hash space, chosen before the counts were
looked at.

Run-separated rather than shot-separated because two shots of one run day share a session leader,
a mini-proposal, a machine configuration and usually a logbook sentence. A shot-wise split would
leave a near-copy of every eval shot in dev and the text channels would be scored on their own
neighbours. `prompts.run` excludes the whole complement from the database *before* searching, so
the hard filter, the BM25 corpus statistics and the k-NN neighbourhoods are computed on one side
only.

**`reference_shot`'s twelve prompts all name shots on the `eval` side, deliberately.** `p145`
196336, `p146` 186194, `p147` 203295, `p148` 194350, `p149` 196399, `p150` 187143, `p151` 186114,
`p152` 194789, `p153` 194953, `p154` 201526, `p155` 196141, `p156` 186389 — 12 of 12 in
`split.eval`. That is not chance (the eval side is 22 % of the corpus; 0.22¹² ≈ 10⁻⁸): they were
chosen with the split in hand, because a reference-shot prompt naming a dev shot is *vacuous* on
`--split eval` — the named shot is excluded before the search. Three things follow, and a reader
is entitled to all three:

* **`reference_shot` on `--split dev` is vacuous.** Every named shot is excluded there. Read that
  category on `--split eval` or `--split all`, never on `dev`.
* `reference_shot`'s coverage and its top-of-table run-day diversity are **not neutral results**.
* 12 of the 200 prompts — 6 % of the coverage numerator — were selected against the split. Every
  other prompt was authored from the vocabulary alone, with no reference to which shots are where.

**What the split landed on, quirks included** (these limit what an eval-split number can mean):

| theme | dev | eval |
| --- | --- | --- |
| `fast_ion_ae` | 6 | **0** |
| `qh_mode` | 24 | **2** |
| `rmp_elm` | 27 | 7 |
| `detachment_divertor` | 55 | 15 |
| `tearing_mhd` | 28 | 9 |
| everything else | 250 | 77 |

The eval side contains **no** `fast_ion_ae` shot and two `qh_mode` shots. That is what an honest
20 % of 232 run days gave, and it is not adjusted: a `fast_ions` result measured on the eval split
is a statement about the lexicon and the retrieval machinery, not about whether DIII-D fast-ion
shots can be found. Category numbers that depend on the corpus rather than on the code should be
read on `--split all`, and the report always says which split it ran on.

**Resolution is not a split number.** `prompts._resolution` calls `phenomena.resolve` on the
prompt text alone — no database, no split — so the per-category resolution column is *identical*
on `dev`, `eval` and `all`. It measures the lexicon against the words physicists type. It is
printed under a split heading only because it shares a table with metrics that are split
numbers; do not read `fast_ions 56.2 %` as a statement about the eval split's 0 fast-ion shots.

## How to extend it

Do not edit the 200 again. v1.1 was the one re-freeze, for defects in the *questions* found by
review before any tuning; the next change is a **new** file — `reference_shot_prompts_v2.csv` — with its own sha256
recorded here and its own test constant, and report both. A number from v1 and a number from v2
are then comparable only when they are labelled, which is the point.

To change the split, the same rule applies: a new `split_v2.yaml`, not an edit. The split is a
pure function of the run id (`eval.prompts.run_bucket`), so any shot list can be cut by it without
touching this file.
