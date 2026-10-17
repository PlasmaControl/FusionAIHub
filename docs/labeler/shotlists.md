# Named shot lists

Status: **added 2026-10-16 (API 11; not committed); renamed to `<event>_<source>`,
`on_demand` and `renamed` added 2026-10-17.** A shot list is a named set of
shots in a file. The review page can open any event on any list, and evaluators score
on lists. Code: `src/labeler/events/shotlists.py`. Routes:
`src/labeler/events/ui/shotlist_routes.py` and `/api/queue` in `app.py`.

## Files

- `data/events/shotlists/<name>.csv` holds a `shot` column. Any other columns
  (batch, order, pool, classes, run_day, note...) are kept and served as they are.
  `<name>.txt` holds one shot per line; `#` starts a comment.
- The name is the file's stem: `[a-z][a-z0-9_]*`. `blind` and `roster` are reserved.
- Naming (owner, 2026-10-17): a dataset list is `<event>_<source>`. The event is a
  short prefix (`ae`, `confine`, `detach`, `elm`, `rwm`, `n1` for mhd_n1, `n2`, `n3`,
  `qmin`, `betap`, `lm`) and the source is the annotator's surname in lower case,
  `chen` for a list the owner's pipeline made. An event with two or more sources
  also has the union `<event>_<a>_<b>`, surnames in alphabetical order. A subset adds
  a suffix (`<event>_<source>_<subset>`). `gold`, `silver`, `silver_batch_*` and
  `catalog` are sets, not sources, and keep their names.
- `<name>.meta.json` (the sidecar) records how the list was made: title, kind, rule,
  derived_from (each input with its sha256), provenance, made_at, made_by, git,
  n_shots, sha256, and blind_dropped when shots were left out.
- `gold` and `silver` stay at `gold/gold_shots.csv` and `silver/silver_shots.csv`
  and keep their stricter readers. They are lists even when no registry exists.
- Shots are integers and unique. **No list may hold a catalog blind test shot**
  (`catalog/cohort.csv`, `blind == True`). When the catalog is present, reading such a
  list fails; `make` refuses to write one, and without the catalog it fails closed.
  Any shot on any readable list can be reviewed on every event, as an unverified row
  when it is off the event's roster (`shotlists.in_any`). That reading rule is what
  keeps the blind shots off the page; a tree without the catalog (a release) trusts
  its lists.
- Drafts: a shot with no source row opens on a silver draft only when the event
  borrows it, meaning a silver or gold shot, or a listed shot off the event's roster.
  A roster shot keeps today's behaviour, and a source pointer at a missing table still
  fails for it. No draft opens on a shot of a `protect: holdout` list, nor when a table
  the release withholds (`hf_export.WITHHELD`, the ELM challenge key) holds the shot
  for the event. The by-shot strips never draw a withheld table.
- `shotlists` is in `events.yaml` `non_category_dirs` and in `hf_export.NON_EVENTS`.
  The lists are not mirrored to Hugging Face yet.

## Registry: `configs/labeler/shotlists.yaml`

```yaml
lists:      # title, kind (curated | derived | dataset), parent, rule, protect, hidden,
            # on_demand, file, sha256, description
  gold: {title: "Gold set", file: gold/gold_shots.csv, protect: test}
  silver_batch_1: {title: "Silver batch 1", parent: silver, hidden: true,
                   rule: "where(silver, batch==1)"}
  catalog: {title: "All shots", kind: dataset, on_demand: true,
            rule: "table(catalog/population.csv)"}
  elm_eldon: {title: "Eldon (ELM challenge key)", kind: dataset, protect: holdout}
events:     # what the page recommends, opens on, and what evaluators score on
  alfven_eigenmode: {recommended: [ae_heidbrink_kaptanoglu, ae_heidbrink, ..., silver, gold],
                     default: ae_heidbrink_kaptanoglu,
                     evaluate: [ae_heidbrink, ae_kaptanoglu, ae_heidbrink_kaptanoglu, gold]}
renamed:    # old list name -> its list, or {event id: list, default: list}
  ae_co2: ae_heidbrink
  cohort: {minimum_safety_factor: qmin_chen, poloidal_beta: betap_chen, default: qmin_chen}
```

- A list file that the registry does not name is still a list. The page shows it
  under "Other lists" with kind `other`.
- A subcategory with no entry uses its parent's entry. An event with no entry
  recommends silver, gold and `catalog` and opens on its roster.
- `hidden: true` keeps a list's file and name but leaves it out of the dropdown and a
  parent's children. A request for it (`/api/queue?list=`) falls back to the event's
  default, with the note "is not offered any more". The five `silver_batch_*` lists
  are hidden: `silver` is one list of 450, and Next walks the selected list end to end.
- `renamed: {old: new}` keeps an old list name working. `shotlists.resolve` (called by
  `entry`, so by `read`, `shots`, `plan`, `make --union/--from`, `show` and every
  route) takes `old` to `new`: page addresses (`#list/<old>/...`), saved views and API
  calls open the new list. `/api/queue?list=<old>` answers with `list` the new name
  and the note "List <old> is now <new>."; the page shows it like any list note. A
  value may be a mapping of event id to list with a `default` (the old `cohort` is
  `qmin_chen` on minimum_safety_factor and `betap_chen` on poloidal_beta); the event
  is the request's, its parent's for a subcategory, else `default`. A name that is
  still a list wins over `renamed`, and `check` reports it, and a list file left under
  an old name. `make` refuses an old name as its target (it names the list to make);
  an old name in `--union` or `--from` is resolved, and the rule and `derived_from`
  record the new one. Dropped lists (`ae_valid`, `ae_valid_fair`) are not in `renamed`:
  asking for one falls back to the default with "no longer exists".
- `on_demand: true` marks a list whose shots mostly have no pre-built review store:
  the server builds one when a shot is opened, like a catalog shot, so holding such a
  shot does not make the status line call it "prebuilt" (`shotlists.on_demand(paths)`;
  `ondemand.protected` decides data root or cache as before). It is set on `catalog`,
  the n=1 lists, `elm_farre`, `elm_eldon_farre`, `confine_butt` and
  `confine_butt_gill` (each under half of the list has a store for its own event).
- `catalog` ("All shots", kind `dataset`) is every shot of the v1 event catalog,
  `data/events/catalog/population.csv` (4,872 shots), less the 50 blind test shots:
  **4,822 shots**. It is the last recommended list of every event. No store is built
  for it ahead of time: opening one of its shots builds the store on demand
  (`on_demand: true`, docs/labeler/on_demand.md), so the status line does not call
  a catalog shot "pre-built". It was made with
  `python -m labeler.events.shotlists make catalog --from-table
  data/events/catalog/population.csv --drop-blind --kind dataset --title "All shots"`;
  the list file is data and is not tracked.
- `default` is the virtual `roster` unless a list equals the event's roster. The AE
  roster is `ae_heidbrink_kaptanoglu`, and the q_min and beta_p rosters are `qmin_chen`
  and `betap_chen`, in the same order.
- `protect: test | holdout` stops `make` from ever overwriting the list.

## CLI

```bash
python -m labeler.events.shotlists list           # every list: size, title, kind, flags
python -m labeler.events.shotlists show NAME      # its entry, sidecar and rows
python -m labeler.events.shotlists make NAME (--from-txt F | --from-table F [--column C]
    | --shots S ... | --union A,B,... | --from LIST) [--where 'batch==2 and pool==corpus']
    [--sort COLUMN] [--title T] [--kind K] [--provenance TEXT] [--drop-blind]
python -m labeler.events.shotlists check          # every list reads; sidecars, inputs and parents agree
```

`make` writes the list and then its sidecar. Each is written to a temporary file and
renamed into place. `--from` keeps the source list's columns. `--union` takes list
names and `table(PATH)` / `txt(PATH)` terms (a source file as `--from-table` /
`--from-txt` read it): `confine_butt` is `union(table(...Jalal_28042024...csv),
txt($LABELER_ROOT/raw_external/jalal_wpqh_2026-05-26_shots.txt))`, and `check`
re-reads each term. `--drop-blind` leaves
blind shots out and counts them; without it, a blind shot is refused. `check` reports a
derived list as stale when an input changed after the list was made. It also reports a
child list holding shots that are not in its parent.

## Routes (API 12)

- `GET /api/shotlists?event=E` returns
  `[{name, title, n, kind, recommended, default, parent, protect, n_saved}]`.
  The recommended lists come first, in the registry's order. The other lists follow,
  each child right after its parent. A child of a recommended list opens the other
  lists, in its parent's order, so the page shows it flat (unindented) at the top of
  "Other lists". The virtual roster entry (`kind: roster`) is last. A list that cannot
  be read is left out, and then the roster becomes the default.
- `GET /api/queue?event=E[&list=N]`: the response has today's keys plus `list` (the
  name shown) and `lists` (the body above).
  - With no `list`, or with `list=roster`, the queue is the roster as before, with the
    gold shots the roster lacks flagged `extra`.
  - With a list, the queue is that list's shots in its order, with shots off the
    roster flagged `extra: true`.
  - For a subcategory, only the list's shots where the parent's saved label marks
    something are included.
  - An unknown or unreadable `N` falls back to the default. `note` then says why, for
    example "List ae_old no longer exists; showing ae_heidbrink_kaptanoglu.". A name that
    `renamed` moved opens its list, with the note "List cohort is now qmin_chen.".
- `GET /api/shots/search?q=<digits>&event=E&limit=20` returns
  `{"shots": [{"shot", "lists", "cached"}]}`: the shots whose number starts with `q`,
  the gold, silver and event-roster shots first, then by number. `lists` names the
  registered, not hidden, lists that hold the shot (`roster` first for the event's
  own); `cached` says whether `E`'s store for it is built and current (any event's with
  no `event`). A full six-digit number that no list holds, and that is not a blind
  test shot, still comes back as `{"shot": n, "lists": [], "cached": false}`: opening
  it builds on demand and succeeds if fdp has data. `q` is one to six ASCII digits
  (else 400 "q must be one to six digits"), `limit` is 1 to 100, an unknown `event` is
  404. A blind shot is never returned.
- `GET /api/shotlist?name=N` returns
  `{name, title, kind, rows: [{shot, ...columns}], meta}`.
- `GET /api/overview?list=N` returns `{list, title, grid_ms, events, shots: [{shot,
  ...columns, saved, total, done}], children: [{name, title, n}]}`. It is
  `/api/silver` for any list. Columns keep their file values, except that `classes`
  is a list, split on `;` as `/api/silver` gives it. It is not paged: it reads every
  shot's progress, which takes about 2 s cold for 572 shots and would take about 18 s
  for a pool-sized list of about 5,000.
- `GET /api/overview?list=N&shot=S` returns `/api/silver/shot`'s events and `t_range`
  for any shot on some list, plus that list's columns for the shot, with `classes` as a
  list.
- `/api/silver`, `/api/silver/shot` and `/api/gold/shots` are unchanged.
- The queue and the list overview are cached until a file they read changes. That
  means the roster, the saves and history, the source table, the taxonomy, the gold
  file, the registry, every list file and sidecar, and the catalog. A file is
  checked by its mtime, inode and size, and a directory also by its listing.

## Evaluation: `shotlists.plan`

`plan(paths, names, population, trained_on, reference, include_trained=False,
scoring="heldout" | "oof")` returns the shots to score on each list. It also returns
the envelope that a record keeps under `sets.<name>`:

```json
{"list": "gold", "title": "Gold set", "list_sha256": "...", "n_listed": 50, "n_scored": 47,
 "missing": {"no_input": [...], "not_in_cv": [...], "no_reference": [...]},
 "trained_on": [...], "scoring": "heldout", "leaky": false}
```

Every listed shot that is not scored is named under `missing` or `trained_on`.
- A shot outside `population` goes to `no_input`, or to `not_in_cv` when scoring is
  out of fold.
- A shot the model trained on is listed under `trained_on`. It is scored only with
  `include_trained`, which also sets `leaky`.
- A shot the reference table lacks goes to `no_reference`.

## Lists (seeds made 2026-10-15 with `make`; renamed and extended 2026-10-17)

| list (was) | shots | from |
|---|---:|---|
| gold, silver (+ silver_batch_1..5, 90 each, hidden) | 50, 450 | their own files; batches `--from silver --where batch==N` |
| catalog (All shots) | 4,822 | `catalog/population.csv` (4,872), blind dropped; made 2026-10-17 |
| qmin_chen, betap_chen (cohort) | 450 each | catalog cohort, not blind, by queue_rank; one per event |
| ae_heidbrink (ae_co2) | 180 | W. W. Heidbrink's CO2 AE annotation (AE format table) |
| ae_kaptanoglu (ae_ece) | 26 | Alan Kaptanoglu's ECE boxes (the owner confirmed the annotator 2026-10-17) |
| ae_heidbrink_kaptanoglu (ae_co2_ece) | 193 | union, the AE roster |
| elm_farre (elm) | 572 | Hiro Farre's ELM label dicts (576 shots, 4 blind left out) |
| elm_eldon (elm_challenge; protect: holdout) | 24 | David Eldon's challenge key |
| elm_eldon_farre (new) | 595 | union (163518 is in both); browsing only |
| rwm_hanson | 33 | Hanson's RWM onsets |
| confine_butt (confinement_jalal, confinement_jalal_wpqh) | 1,113 | Jalal Butt: his regime table of 28 April 2024 (432) and his WPQH archive of 26 May 2026 (769), 88 in both, 5 blind left out |
| confine_gill (new) | 424 | Kevin Gill's confinement workbook (426 shots, 2 blind left out) |
| confine_butt_gill (new) | 1,113 | union; every workbook shot is already in Butt's lists |
| detach_chen (detach_tangtv) | 41 | the owner's plasma_tv TangTV inversions |
| n1_seo (new) | 8,493 | Jaemin Seo's n=1 tearing archive `/projects/EKOLEMEN/tm_data` (8,505 shots, 12 blind left out) |
| n1_farre (new) | 28,488 | Hiro Farre's n=1 tearing labels `survival_tm_2/tm_labels.tar` (28,521 shots, 33 blind left out) |
| n1_farre_seo (new) | 28,494 | union (8,487 shots are in both) |

`ae_valid` (60) and `ae_valid_fair` (19) were dropped 2026-10-17; the 19 fair shots
stay in `outputs/labeler/ae/baselines/evaluation.json` `shots.fair`.

**The n=1 lists.** There are two label sets, not one. Seo's is the training archive of
his n=1 tearing CNN (`z.npy` the shot of each of 639,555 samples, `y.npy` column 1 the
label, 1,801 positive shots). Farre's is `survival_tm_2`'s `tm_labels.tar`, one
`<shot>_ntm.npy` (a 1 ms 0/1 series) for each of 28,521 shots, 4,820 with a positive
label; they are the shots of `neoclassical_tearing_mode_format_2026_v1.csv`.
`tm_labels.h5` is the same labels at 20 ms for 18,064 of the tar's shots, so it is
not a third set. Seo's shots are nearly all in Farre's (8,499 of 8,505), and 1,671 of
Seo's positives are also Farre's positives. A list holds every labelled shot, positive
or not; the positive shots alone are 1,801 (Seo) and 4,820 (Farre). Only 824 (Seo) and
2,236 (Farre) of the shots are in the catalog, 2,237 together (46 % of its 4,822), so the
n=1 lists are `on_demand`. The by-shot overview (`/api/overview?list=`) reads every
shot's progress and is not meant for `n1_farre`; its queue walks it fine.
