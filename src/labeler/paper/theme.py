"""The paper's colour theme: sea green first, coral second.

The owner, 2026-10-14: "primary colors be more sea green and secondary coral ...
for all figures. like if it's rainbow then ofc keep it having different colors.
but wanted to give the paper more of a theme". So wherever a figure pairs two
things, the first (Tokamak-SI, ours, the main class) is `SEA_GREEN` and the
second (the original labels, a published model, the other class) is `CORAL`; a
figure with more classes than that keeps them apart with the extras below,
chosen to sit with the pair.
"""

from __future__ import annotations

SEA_GREEN = "#2E8B57"
CORAL = "#FF7F50"
#: A darker sea green: the reviewed subset of a sea-green set, a line over a
#: sea-green shade.
SEA_GREEN_DARK = "#123D28"
#: The pair's light tints, for a second shade of the same class and for spans.
SEA_GREEN_LIGHT = "#A8D5BA"
CORAL_LIGHT = "#FFD2C2"
#: A light mint, for lines drawn over a dark image (magma, black).
MINT = "#9ADBB8"
#: The extra classes of a categorical set, in the order to use them.
PLUM = "#CC79A7"
SLATE = "#3D5A80"
SAFFRON = "#E9C46A"
SKY = "#8ECAE6"
