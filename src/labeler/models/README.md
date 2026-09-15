# Model roster

One folder per trained model. A folder is `<slug>/` with:

| file | contents |
|---|---|
| `README.md` | HuggingFace-style model card. YAML front matter plus a custom `labelmaker:` block; the front matter is what `registry.py` parses. |
| `spec.py` | the only per-model code: name mapping, lag structure, training domain, output columns, loader. Exposes `ADAPTER`. |
| `__init__.py` | empty. |

## Naming

`d3d_tearing_time_to_event_dsm_continued` is the `[_<variant>]` case of the slug
grammar below: the same architecture, inputs and outputs as the model it names,
with weights that continue that model's under-trained fit
(`scripts/labeler/retrain_tearing_dsm.py`). Its `spec.py` imports the base
spec's pieces rather than copying them.

Slug grammar: `<device>_<phenomenon>_<what-is-predicted>_<architecture>[_<variant>]`,
lower snake case for the folder and Python module. The card id is the same slug
hyphenated under the `plasmacontrol/` namespace, which is what a HuggingFace repo
would be called. Label groups in `<shot>_labels.h5` use the folder name.

| Folder | Card id | Task | Status |
|---|---|---|---|
| `d3d_tearing_onset_cnn1d` | `plasmacontrol/d3d-tearing-onset-cnn1d` | binary + regression | implemented |
| `d3d_elm_time_to_event_dsm` | `plasmacontrol/d3d-elm-time-to-event-dsm` | survival (ELM risk at 4 horizons) | implemented |
| `d3d_tearing_time_to_event_dsm` | `plasmacontrol/d3d-tearing-time-to-event-dsm` | survival (risk at 3 horizons) | implemented |
| `d3d_tearing_time_to_event_dsm_continued` | `plasmacontrol/d3d-tearing-time-to-event-dsm-continued` | survival (risk at 3 horizons), retrained | implemented |
| `d3d_ech_beam_fate_mlp` | `plasmacontrol/d3d-ech-beam-fate-mlp` | 3-class + regression | scaffold |
| `d3d_ech_deposition_torbeamnn` | `plasmacontrol/d3d-ech-deposition-torbeamnn` | regression | scaffold |
| `d3d_kinetic_equilibrium_rtcakenn` | `plasmacontrol/d3d-kinetic-equilibrium-rtcakenn` | profile regression | scaffold |
| `d3d_inpa_image_cnn` | `plasmacontrol/d3d-inpa-image-cnn` | image regression | scaffold |
| `d3d_ae_activity_seldnet` | `plasmacontrol/d3d-ae-activity-seldnet` | binary + regression | implemented |

`status: scaffold` means the folder documents a model that labeler cannot run
yet; its card's `blocked_on` list says exactly what is missing, and importing its
`spec.py` raises `NotImplementedError`.

Two models carry weights labeler **fitted itself**.
`d3d_elm_time_to_event_dsm` is one: upstream's Keras graphs need 64 BES inputs
the FAITH corpus fills on 2 of 24 sampled shots, so labeler refitted the same
architecture on upstream's own rows using only the 60 columns that are not BES -
which, measured, is the better model of the two (see its card).

`d3d_ae_activity_seldnet` is the other, and the one whose architecture is ours
too - no upstream artifact answers "is an Alfven eigenmode present now, and at
what frequency". Its network is `src/labeler/ae/model.py`, its label construction
`src/labeler/ae/labels.py`, its training `scripts/labeler/ae_train.py`,
and its design is `docs/superpowers/specs/2026-09-05-labelmaker-phase3-design.md`
section 5. It is also the only model whose input is a **waveform**: the raw
4-chord CO2 record at 500 kHz, kept at its native rate through the feature layer
and transformed by the adapter itself.

## Deliberately excluded

Recorded so they are not re-added by mistake:

- **TokEye** (`tokeye_unet`, `big_tf_unet`) and **`ae_tf_maskrcnn`** -
  spectrogram-to-pixel-mask models. A different I/O contract (image in, mask
  out) and they already ship as their own packaged application. This excludes
  them as *label producers*, not as tools: `big_tf_unet`'s coherent-mode channel
  is used offline to build the AE model's training labels
  (phase3-design section 5.5). It never publishes a label of its own, so the
  exclusion stands.
- **TokaMind** (`tokamind_base_v2`) - MAST-pretrained; its tokenizer and inverse
  decode live outside the saved graph.
- **diag2diag** - excluded at the project owner's direction. No technical reason
  was recorded, so do not infer one; ask before re-adding it.
- Anything from `tokamak_deploy_bench`'s `models/` directory. That repo is a
  latency benchmark: it feeds random noise to models and stores only timings. It
  is a useful *index* of what exists (`MODEL_ROSTER.md`), never a source of
  weights. labeler loads every model from its upstream source of truth.
