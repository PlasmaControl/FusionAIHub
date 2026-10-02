# Group versus individual event annotations

This note treats `iscrowd` as an annotation/evaluation policy, not as a
learned event class. It is relevant when an ELM burst train, a sawtooth
sequence, or another unresolved event span is labeled as one group while
some events in the span have individual times or masks.

## What COCO `iscrowd` means

COCO does not define a `crowd` category. `iscrowd` is a per-annotation flag
(`0` by default) saying that the annotated region is a *crowd region*: a
group represented by one region rather than separately delineated instances.
The official mask API changes IoU for such a ground truth from intersection /
union to intersection / detection area, allowing a detection that is a
subregion of the group to match it. See the comments and implementation in
[`mask.py`](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/mask.py)
and [`_mask.pyx`](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/_mask.pyx).

In the official Python evaluator, crowd ground truths are marked `ignore`;
multiple detections can match the same crowd region, and detections matched
to an ignored ground truth are excluded from precision/recall accumulation.
The relevant matching and `dtIgnore` logic is in
[`cocoeval.py`](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).
Thus `iscrowd=1` is neither a second class nor a positive instance target.
It is also distinct from an ordinary instance that happens to overlap
another instance.

Detectron2's COCO loader preserves the flag, its default dataset filtering
removes images having only crowd annotations, and its standard
`DatasetMapper` drops `iscrowd=1` annotations before constructing training
instances. See the [dataset loader](https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/coco.py),
[training-data filtering](https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/build.py),
and [mapper](https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/dataset_mapper.py).
This establishes Detectron2's default behavior, not a universal rule for
every framework. A custom mapper/loss is required to use a group region as
an ignore mask; a naive binary conversion can instead make unresolved group
pixels background. No existing FAITH model becomes crowd-aware merely by
adding an `iscrowd` field.

## Implications for FAITH's current models

* [`events/unet.py`](../../src/labeler/events/unet.py) is a vendored TokEye
  U-Net with two class-agnostic channels: coherent activity and transient
  activity. Its transient channel is a natural proposal mask for ELMs and
  sawtooth crashes, but has no instance identity, count, or group flag.
* [`ae/seg/model.py`](../../src/labeler/ae/seg/model.py) is a small binary
  SegNet over time-frequency rows. Its training loss in
  [`ae/seg/train.py`](../../src/labeler/ae/seg/train.py) already supports
  `IGNORE` pixels. This is the right primitive for unresolved regions: use
  unknown/group-only pixels as ignore for individual-instance loss rather
  than labeling them background.
* Public event labels are interval tables (`t_start`, `t_end`, confidence),
  not COCO instances. The existing ELM DSM is a horizon risk model, so it is
  useful for event probability but does not solve interval boundaries,
  multiplicity, or group-versus-individual semantics.

## Model choices

For a first research baseline, retain the existing class-agnostic U-Net or
SegNet as a dense evidence head and add a one-dimensional temporal head over
its time-pooled features. The baseline should track semantic/event and
group-envelope masks, while applying separate individual boundary/count/
identity losses only where individual labels exist. Predict, per event type,
(a) eventness/activity, (b) start and end boundary distributions or interval
proposals, and (c) a group envelope/confidence. A group annotation supervises
(a) and the envelope while masking (b), count, and identity losses. An
individual annotation supervises all available fields. This supports other
phenomena without pretending that unresolved bursts have known count or
fabricated centers.

If annotations become sufficiently complete, compare two stronger families:

* **Set prediction:** DETR's learned queries and bipartite matching provide a
  principled variable number of interval/mask predictions
  ([paper](https://arxiv.org/abs/2005.12872)). For one-dimensional events,
  replace boxes with `(start, end)` (or center/width) and use temporal IoU in
  matching. Add an explicit group query/output only when group labels are
  real targets; do not match a group envelope one-to-one against every hidden
  constituent.
* **Mask/query segmentation:** Mask R-CNN is a strong per-instance baseline
  ([paper](https://arxiv.org/abs/1703.06870)); Mask2Former supports semantic,
  instance, panoptic, and video masks using mask classification and masked
  attention ([paper](https://arxiv.org/abs/2112.01527),
  [video extension](https://arxiv.org/abs/2112.10764)). For spectrograms,
  its 2-D masks can represent time-frequency regions; for 1-D signals, query
  masks over time are a simpler analogue. These models require careful
  target matching and enough individual labels to justify instance training.

For weak or point labels, temporal action-localization methods are relevant
as training ideas, rather than direct drop-in models. ActionFormer directly
predicts temporal action moments without predefined proposals
([ECCV paper](https://www.ecva.net/papers/eccv_2022/papers/ECCV/papers/136640485.pdf));
point-supervised proposal methods generate flexible-duration proposals and
pseudo-labels from sparse temporal points
([PTAL paper](https://arxiv.org/abs/2310.05511)). Their pseudo-label expansion
must be constrained by physics, signal evidence, and annotation uncertainty;
thresholding an activity trace alone can split one group or merge distinct
events.

## Recommended target representation

Keep annotation facts separate:

* `individual`: a resolved event with an interval and, where applicable, a
  time-frequency mask or ridge;
* `group`: one envelope interval/mask whose constituent count and individual
  boundaries are unresolved;
* `unknown/ignore`: not enough evidence to say whether the region is event or
  background.

For each group, store an optional lower/upper count bound, but never create
synthetic individual timestamps merely to satisfy an instance model. The
repository's compatible representation is an optional CSV `attrs` JSON field
with `iscrowd: 0` for resolved individual and `iscrowd: 1` for a group,
aligned with the COCO-style API; `None` means unspecified. Legacy rows with
no `attrs.iscrowd` are therefore unspecified, not implicitly COCO `0`.
In a spectrogram, a group mask can be a union/envelope and individual masks
can overlap it. If a training tensor must be dense, use separate channels for
known-positive envelope, known-negative background, and ignore/uncertain; do
not encode unresolved group pixels as individual positives or negatives.

The editor permits individual intervals inside an overlapping crowd envelope.
In the proposed objectives, those individuals retain their classification,
boundary and identity targets. The group-only ignore region is the part of the
envelope without resolved individual supervision. The envelope remains a group
target, and its total count remains unresolved even when some constituents have
been delineated.

## Loss and evaluation

Use masked losses: individual classification, boundary, count, and identity
terms only on resolved individuals; envelope segmentation and group presence
terms on group labels; no loss on unknown regions. For query models, exclude
group-only targets from one-to-one individual matching, or implement a
many-to-one group match with an overlap/containment objective. The latter is a
proposed extension, not COCO behavior.

Report strata separately rather than one pooled score:

1. group/envelope presence precision-recall and interval temporal IoU or
   coverage (does the predicted envelope cover the labeled span without
   excessive spillover?);
2. resolved individual event precision/recall/F1 at onset tolerance bands,
   boundary error, and temporal IoU/AP;
3. count error or count interval coverage only on groups with reviewed count
   bounds;
4. time-frequency mask Dice/IoU for resolved masks, plus envelope coverage
   and false-positive area for group masks;
5. calibration and abstention/ignore coverage, split by event type (ELM,
   sawtooth, and other events), shot, diagnostic, and annotation resolution.

Do not use ordinary COCO AP with `iscrowd` as if it solved this problem: COCO
crowd matching intentionally ignores the matched detections and does not
measure recovery of hidden instance count. Likewise, an ordinary category
or frame-level score measures semantic/event presence, not instance count,
onset identity, or individual interval quality. If a COCO-like evaluator is
useful, implement a documented temporal analogue and report its individual
and group tracks separately.

## Limitations and research risks

The cited architectures assume their task's annotation semantics; none
establish that ELMs or sawtooth have separable visual instances in every
diagnostic. TokEye's transient mask is evidence, not ground truth. Group
envelopes can include overlapping bursts, diagnostic saturation, or unrelated
transients. Evaluation must preserve shot-level splits and annotation
provenance, and should include an abstain/unknown path so apparent gains do
not come from converting unresolved spans into convenient negatives.
