# Figure 1 fix history (index)

The superseded per-round reports, plans and captions no longer live in the tree. Git
holds them; the current status is `outputs/labeler/paper/fig_interpreter_tokeye/report.md`.

To read an old round, use the commit that recorded its renders and audit:

| Round | Record commit | Read it with |
|---|---|---|
| 5 | `d94df795` | `git show d94df795:outputs/labeler/paper/fig_interpreter_tokeye/fix_round5.md` |
| 6 | `a09fcf00` | `git show a09fcf00:outputs/labeler/paper/fig_interpreter_tokeye/fix_round6.md` |
| 7 | `6c57e279` | `git show 6c57e279:outputs/labeler/paper/fig_interpreter_tokeye/report.md` |
| 8 | `41ba2067` | `git show 41ba2067:outputs/labeler/paper/fig_interpreter_tokeye/report.md` |
| 9 | `1fdf3762` | `git show 1fdf3762:outputs/labeler/paper/fig_interpreter_tokeye/report.md` |
| 10 | `3941c5b6` | `git show 3941c5b6:outputs/labeler/paper/fig_interpreter_tokeye/report.md` |

The 757-line consolidated history that this file used to hold (rounds 1 to 8 as separate
reports) is `git show 1fdf3762:docs/labeler/fig1_fix_history.md`. Rounds 5 and 6 kept
their audit as `fix_round5_audit.json` and `fix_round6_audit.json` in the same folder;
from round 7 on it is `audit.json`.

**Owner revision (r4-fig1b).** After round 11 the owner asked for the original Figure 1
shot and a linear frequency axis. Figure 1 is now shot 199563, 700-5800 ms, on one linear
0-250 kHz axis (no 30/60 kHz scale breaks); its records, audit and README are in
`outputs/labeler/paper/fig_interpreter_tokeye/fig1b/`. The rounds above describe the
earlier six-shot, three-scale figure.
