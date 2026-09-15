"""The evaluation harness: the frozen prompt set, the latency table, phenomenon recall.

Three rules hold across this package, and they are the reason it exists at all:

* **The evalset is frozen before anything is tuned.** `configs/shot_design/evalsets/` is authored
  once and its sha256 is asserted by `tests/shot_design/test_evalset_frozen.py`. A retrieval change
  that makes a number move is a change to retrieval; a prompt edit that makes a number move is
  unmeasurable, so the second is locked out.
* **A miss is a finding.** Nothing here decides what "good" is beyond the two bars the plan
  states (coverage >= 95 %, phenomenon resolution >= 80 % on qh_mode / elm_rmp / fast_ions, and
  the Appendix B latency budgets). Everything else is reported as the number it is.
* **A proxy says it is a proxy.** The 20 hand-graded prompts have a human rubric in the evalset
  and a machine-checkable stand-in here, and `ProxyGrade` carries the difference in its own
  `caveat` field so no reader can mistake one for the other.

Out of scope here, deliberately: the *physical* quality of the detectors themselves. Agreement
with a TokEye or teacher label is not physical accuracy, and establishing the latter is the
labeler workstream's task, not this harness's.
"""

from __future__ import annotations

__all__ = ["latency", "phenomenon_recall", "prompts"]
