# Figure 1 — Fix round 5

Status: **DONE_WITH_CONCERNS**. Every Important and Minor item in
`fig1-opus5.md` and `fig1-sol5.md`, together with the owner's fifth-round
instructions, is addressed. The concern is the pending sawtooth-stream
regeneration; this round consumes the newest complete source currently available.

Commit range for this round: **2ae5b1fa..HEAD**, branch **r4-fig1**. Presentation
and source helpers: **cf046f13**; cadence and eligible-pixel component counts:
**21e72826**; six clean-render records/captions: **2965d609**. The final commit
adds the audit and this report. All commits have `labeler:` subjects and the
requested `Co-Authored-By: Codex gpt-6.1-sol <noreply@openai.com>` trailer.

### Presentation and source changes

`fig_interpreter_tokeye.py` gives **0–30 kHz** its own **1.30-unit** panel.
Its measured height is **0.856495 in** for the primary and **0.815930 in**
for the sawtooth alternate. The processed 30–55 kHz band is omitted and the
raw 0–55 kHz panel remains; the break and caption disclose this choice. Smaller
high-frequency/raw bands and keys in the empty right margin pay for the enlarged
n panel. The overall size stays **6.75 × 5.5 in**, with fonts **≥7 pt**.
Source: `fix_round5_audit.json`, `renders[].layout` and `pdf_size_in`.

NTM uses a **one-print-pixel inner edge, alpha 0.45**. Both the component's
unique dominant n and every marked pixel must be **n∈{1,2}**; the measured view
limits outlines to **≤30 kHz**. Components with only n=3 pixels during the
NTM interval are no longer counted as tagged. This corrects the primary's
NTM count to **11**, and 186636's to **4**. The n colours remain readable.
The keys now read **“NTM (n=1/2, ≤30 kHz)”** and
**“AE (detector band ≥80 kHz)”**; the latter wraps to fit the margin.
The raw resolution note, AE callout and n=1/NTM callout sit outside the data
with neutral grey leaders. Source: each record's `filter`, `drawn.layout`,
`drawn.blobs.tagged` and `drawn.ntm_measured_pixel_audit`.

The sawtooth row is shown only if the source has positive-duration PRESENT
time in the displayed window, for every shot. The primary has none, so its row
and empty crash strip are omitted. Its exact source states are retained in JSON:
**832.099943 ms uncertain**, **967.900057 ms unassessed**, **0 ms present**.
Only **201973** has a displayed row in these six windows, with **66.099998 ms
present** and **3** ECE-verified, ELM-vetoed crash ticks. Sawtooth uses vermillion
**#D55E00**, distinct from the yellow n=3. The bottom “present” chip contains
only event colours actually drawn as present bars. Sources:
`fix_round5_audit.json`, `sawtooth_state_duration_ms`, `verified_crashes`,
`sawtooth_track_shown`, and each `layout.present_chip_colours`.

`figure_sources.sawtooth_display` coalesces touching equal states and merges
state slivers **<10 ms** into the longer touching neighbour, processing shortest
first and preferring the earlier neighbour on ties. It never bridges gaps.
This is a display operation: source intervals/categories and exact crash times
stay unchanged. Each record retains raw `state_intervals_ms`, plus
`display_intervals_ms` and `display_merge` with its rule, threshold and all
changes. The current displayed 201973 row needs no sliver reassignment; the
primary's slivers disappear with the omitted row. Regression tests exercise
an actual short-state merge, source immutability, boundary visibility and gaps.

Late-line bands now come from eligible coherent components after the last
PRESENT AE interval, during ABSENT AE time, in the detector's input band.
Each component must persist **≥150 ms** continuously. Frequency bounds come
from its pixels. The primary's bounds are **109.863286–188.476571 kHz**,
captioned as **110–188 kHz**; 201973's are **177.246087–220.703117 kHz**.
The other four windows have no qualifying late-line description; in particular,
186636's previous window-edge sliver no longer triggers it. Sources:
`fix_round5_audit.json`, `late_untagged_high_frequency` for each render, generated
by `figure_sources.late_untagged_lines`. No 170–250 kHz scientific band remains
hard-coded in the renderer or caption.

All event keys and caption claims follow displayed content and source tier.
191376/191782 have no AE/NTM keys or highlight claims and correctly name ELMs
as detector output; neither has the expert-circle key. The current captions
have **26–143 words**. The primary clause says the AE tint follows the
detector's **25 ms bins**; the earlier frame-model alternate uses its actual
**10 ms** cadence, recorded as `temporal_bin_ms`. “Bt not in the local corpus”
replaces the misleading missing-measurement wording when the displayed sawtooth
assessment needs that qualification. The frame-model jargon sentence is gone
from every caption. Sources: six caption files, records and audit `caption_words`.

Confinement intervals now retain numeric `category` alongside the track's own
state/regime names. Curated category **2** is **L-mode**, category **3** is
**QH-mode**, and category **5** is **uncertain**. The D-alpha fallback retains
its H-mode/binary mapping, including numeric category **5**, without claiming
a curated L-mode label. Thus 191376's L-mode interval is no longer serialized
as “uncertain”. Source: all six `tracks.confinement.states` and
`state_intervals_ms`; audit `confinement_intervals_ms` checks that mapping.

### Data, outputs and final counts

Shot selection, thresholds, model inputs and the reviewed windows are retained.
201978 remains the primary for its clear AE cascade and persistent low-frequency
n structure; 201973 is the strongest visual alternate, 203187 another cascade
example, and 186636 adds imported tearing/expert ELM provenance. 191376/191782
remain diagnostic/regime comparisons, not complete substitutes for the primary.
The cohort is unchanged: **four train and two validation** shots, **zero blind
test** shots. These are descriptive coincidences, not classification scores.
All inputs were local/read-only; no fetching, training or threshold tuning ran.

The newest complete source remains
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/saw/fix2/shots`, with
**500 requested/500 processed** shots, no errors, and completion SHA-256
`79a2c18ed4aae29ba2f08aa8caaeff0c74c23cf7a24b4b150b6aa943500f10c4`.
The live completion and six shot hashes were rechecked and still match the
immutable figure-owned snapshot recorded in `sawtooth_source_manifest.json`:
`round4/fig1/saw_source/fix2-79a2c18ed4aa/shots`. Every final render and rebuild
uses that snapshot through **--sawtooth-source**. The sawtooth stream is still
regenerating; its newer incomplete output was not substituted. The controller
can rerender with its eventual complete output without changing renderer code.

Source for every count and window below: **fix_round5_audit.json**, generated by
the committed `scripts/labeler/paper/fig1_audit.py`, plus the six underlying
`<shot>.json` records in `outputs/labeler/paper/fig_interpreter_tokeye/`.

| Shot | Split | Window (ms) | AE components | NTM components | D-alpha triangles | Sawtooth row | Caption words |
|---|---|---|---|---|---|---|---|
| 201978 | train | 1500–3300 | 319 | 11 | 49 | omitted | 104 |
| 201973 | val | 1600–3350 | 154 | 14 | 65 | shown | 143 |
| 203187 | train | 1700–3150 | 262 | 8 | 6 | omitted | 72 |
| 186636 | val | 1300–3900 | 307 | 4 | 0 | omitted | 70 |
| 191376 | train | 1500–2900 | 0 | 0 | 2 | omitted | 32 |
| 191782 | train | 1800–3700 | 0 | 0 | 0 | omitted | 26 |

Large outputs under `/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/`:
the primary `fig_interpreter.pdf`, `.png`, `.json`, `caption.tex`, and the same
four files in `alt_201973/`, `alt_203187/`, `alt_186636/`, `alt_191376/`,
`alt_191782/`. PDFs retain vector text/lines; PNGs are **150 dpi**.
Logs: `render5_<shot>.log`, `tests5.log`, `lint5.log`, `format5.log`, `audit5.log`.
The external and committed `fix_round5_audit.json` are identical. Small committed
outputs are the six records/captions, audit, plan, README and `fix_round5.md`.

All six final PNGs were opened and inspected: enlarged n ridges, visible n=3,
translucent NTM edges, off-data labels, keys/source text, ELM provenance and
sawtooth row visibility. All trial PNGs and the rebuilt primary PNG were also
opened. The audit checks every heading/key text bound lies inside the page.

### Draft caption

DIII-D shot 201978. Raw bands normalised separately; TokEye's U-Net extracts
coherent modes. Processed 30–55 kHz omitted. AE: neural interferometer detector
(p≥0.5; training targets used TokEye's mask); NTM: magnetic detector (unverified;
shared inputs); H-mode: D-alpha detector; ELMs: expert. Pink: AE time overlap in
detector band ≥80 kHz; tint follows the detector's 25 ms bins. n measured ≤30 kHz.
NTM outlines require dominant and pixel n=1/2. The n=2 ridge near 18 kHz is
consistent with a second harmonic of n=1. Late 110–188 kHz lines stay untagged
where AE detector is absent. circles: expert ELM interval (one span for many
ELMs); triangles: threshold D-alpha peaks (not annotated).

This is the primary's generated caption, copied from `201978.caption.tex`
without TeX wrapping; the controller can edit it for the manuscript. The source
JSON retains the frequency-ratio support behind the harmonic sentence.

### Deviations from the brief

**Sawtooth crashes are not projected onto rotating modes.** This explicitly
deviates from the owner's original request to project sawtooth detections below
60 kHz and, if present, illustrate sawtooth seeding an NTM. A crash is a point
event; overlap with an n=1 mode does not identify that mode as its precursor.
The available crash evidence does not establish the precursor or a causal
seeding relation. The figure therefore uses independently ECE-verified,
ELM-vetoed crash ticks, while the physics-state row is conditional on actual
PRESENT time. No seeding claim is fabricated for the primary, whose source
window has no PRESENT sawtooth time. `mode_tags.tag_blobs` excludes sawtooth
from mode tags; exact physics intervals and tick evidence remain in the records.

AE projection begins at **80 kHz**, rather than the original brief's 60 kHz,
because 80–250 kHz is the detector's input band. White 60–80 kHz structures are
outside that input band. The processed 30–55 kHz band is omitted to enlarge the
measured-n panel, as explicitly authorized in round five. The raw band remains.
The catalog sawtooth frame model is excluded: its output is not corroborated
crash/precursor evidence; that explanation belongs in this report/text, not in
the caption. The existing complete sawtooth snapshot is retained pending the
new source's completion, as authorized by the current rerender instruction.

### Verification and remaining work

Renderer records were generated externally from clean committed **21e72826**;
all six have `render_started_from_clean_head: true`, matching code hashes,
and the same source revision. After record commit **2965d609**, the final audit
and scratch rebuild ran from another clean committed HEAD. Nothing was pushed,
merged, rebased, written to the main checkout, or changed in the manuscript.

Covering tests, with required TMPDIR and local-data variables, through the
mandated wrapper:

```text
bash /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/scratch/bin/pt.sh
  /scratch/gpfs/nc1514/FusionAIHub-r4-fig1
  tests/labeler/test_paper_figure_sources.py
  tests/labeler/test_paper_mode_tags.py tests/labeler/test_paper_label_figure.py
  -q -p no:cacheprovider
59 passed in 4.08s
```

Ruff check on the six changed Python files: **All checks passed!**
Ruff format --check: **6 files already formatted**. `git diff --check` passes.
New failure cases were observed before fixing n=3 pixels, regime serialization,
short-state display merging, conditional captions, late-line bounds/duration,
source cadence and components whose eligible interval contains only n=3.
No full suite ran, per the implementer rules. The mandated tmpsweep ran after
rendering and after the audit/rebuild.

```text
pixi run --frozen --no-install --manifest-path
  /scratch/gpfs/nc1514/FusionAIHub/pyproject.toml -e labelmaker
  python scripts/labeler/paper/fig1_audit.py --rebuild-primary
  --out /scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/fig1/fix_round5_audit.json
Audited 6 non-blind renders; zero projection violations
Primary PDF and PNG rebuild identically byte-for-byte
```

Every projection audit has **zero** outside-PRESENT/band pixels; every NTM audit
has **zero** unmeasured, above-30-kHz or measured-n=3 outline pixels. Sources,
committed renderer hashes, figure hashes, regime categories, raw physics states,
display rows, conditional keys, caption lengths and external/committed copies
all match. The primary PDF and PNG hashes are identical before and after rebuild;
full hashes and the rebuild command are in `fix_round5_audit.json.reproducibility`.

No unresolved fifth-round implementation finding remains. The next action is
the controller's rerender after the sawtooth stream completes, followed by the
same audit and PNG inspection. Scientific limitations remain disclosed: NTM
is unverified, AE targets depend on TokEye, harmonic consistency does not identify
an island, and the density proxy limits local ECE assessment without proving
cutoff. A co-occurrence does not establish a precursor or causation.
