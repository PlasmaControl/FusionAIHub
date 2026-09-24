"""The DIII-D event catalog: every cohort shot assessed for six phenomena.

- `states`: the four states and what each phenomenon records;
- `points`: the point-event table (disruption times, ELM and crash times);
- `check`: the checks a catalog table must pass before anything is written;
- `window`: each shot's assessed window and Ip flat-top, from high-rate Ip;
- `population`: the corpus shots that pass the four inclusion rules;
- `cohort`: the weighted draw, its splits, its review queue and its manifest.
"""
