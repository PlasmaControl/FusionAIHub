# Model roster

One folder per trained model. A folder is `<slug>/` with:

| file | contents |
|---|---|
| `README.md` | HuggingFace-style model card. YAML front matter plus a custom `labelmaker:` block; the front matter is what `registry.py` parses. |
| `spec.py` | the only per-model code: name mapping, lag structure, training domain, output columns, loader. Exposes `ADAPTER`. |
| `__init__.py` | empty. |

## Naming

Slug grammar: `<device>_<phenomenon>_<what-is-predicted>_<architecture>[_<variant>]`,
lower snake case for the folder and Python module. The card id is the same slug
hyphenated under the `plasmacontrol/` namespace, which is what a HuggingFace repo
would be called. Label groups in `<shot>_labels.h5` use the folder name.

| Folder | Card id | Task | Status |
|---|---|---|---|
| `d3d_tearing_onset_cnn1d` | `plasmacontrol/d3d-tearing-onset-cnn1d` | binary + regression | implemented |
| `d3d_elm_time_to_event_dsm` | `plasmacontrol/d3d-elm-time-to-event-dsm` | survival | scaffold |
| `d3d_tearing_time_to_event_dsm` | `plasmacontrol/d3d-tearing-time-to-event-dsm` | survival (risk at 3 horizons) | implemented |
| `d3d_ech_beam_fate_mlp` | `plasmacontrol/d3d-ech-beam-fate-mlp` | 3-class + regression | scaffold |
| `d3d_ech_deposition_torbeamnn` | `plasmacontrol/d3d-ech-deposition-torbeamnn` | regression | scaffold |
| `d3d_kinetic_equilibrium_rtcakenn` | `plasmacontrol/d3d-kinetic-equilibrium-rtcakenn` | profile regression | scaffold |
| `d3d_inpa_image_cnn` | `plasmacontrol/d3d-inpa-image-cnn` | image regression | scaffold |
| `d3d_ae_activity_seldnet` | `plasmacontrol/d3d-ae-activity-seldnet` | binary + regression | planned |

`status: scaffold` means the folder documents a model that labelmaker cannot run
yet; its card's `blocked_on` list says exactly what is missing, and importing its
`spec.py` raises `NotImplementedError`.

`planned` is weaker still: there is no folder yet. `d3d_ae_activity_seldnet` is
the one model labelmaker will **train itself** - no upstream artifact answers
"is an Alfven eigenmode present now, and at what frequency" - so its
requirements live in `docs/superpowers/specs/2026-09-05-labelmaker-phase3-design.md`
section 5 until the training data and label construction are settled.

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
  weights. Labelmaker loads every model from its upstream source of truth.
