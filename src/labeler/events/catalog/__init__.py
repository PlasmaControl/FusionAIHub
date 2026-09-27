"""The DIII-D event catalog: cohort shots selected to be assessed for six phenomena.

Labels come in later parts. The population passes five inclusion rules; the
cohort applies rule 5 from `runaway.csv` after the population module's rules 1-4.

- `states`: the four states and what each phenomenon records;
- `points`: the point-event table (disruption times, ELM and crash times);
- `check`: the checks a catalog table must pass before anything is written;
- `window`: each shot's assessed window and Ip flat-top, from high-rate Ip;
- `population`: the corpus shots that pass inclusion rules 1-4;
- `runaway`: the Thomson scan used by the cohort to apply rule 5;
- `cohort`: the weighted draw, its splits, its review queue and its manifest.
"""
