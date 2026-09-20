### Task D3: `shot_design simulate` CLI

**Files:**
- Create: `src/shot_design/simulate/cli.py`
- Modify: `src/shot_design/cli.py` (register `simulate` subcommand → `simulate.cli.run(args)`)
- Test: `tests/shot_design/test_simulate_cli.py`

**Interfaces:**
- `python -m shot_design simulate <design-ident> [--seed 0] [--k0 20] [--n-predict 80] [--decode filterscopes,mhr,mirnov,ts_core_density,ts_core_temp] [--device cuda|cpu] [--out DIR]`
- Default out: `paths.data_root / "outputs" / <ident> / "simulation"`; writes `status.json` `{"state": "running|complete|failed", "started", "finished", "error", "report": "report.md"}` first thing and last thing.
- Steps: `program.load_program(ident)` → `program.export_ignite(program, paths)` (design seed) → `program_reference.reference(program.reference_shot, paths).cache` (real) → `core.load_dynamics` → `core.actuator_arms` → `core.run_paired` → `decode` → `report.write`.

- [ ] **Step 1: Failing test** — monkeypatch `core.load_dynamics`, `core.run_paired`, `decode.decode_modalities`, `program.load_program`, `program.export_ignite`, `program_reference.reference` with small fakes; assert `status.json` goes `running → complete`, `report.md` exists, and a raising fake leaves `status.json` `failed` with the error text and exit code 1.

- [ ] **Step 2–4:** fail → implement → pass.

- [ ] **Step 5: Commit**

```bash
git add src/shot_design/simulate/cli.py src/shot_design/cli.py tests/shot_design/test_simulate_cli.py
git commit -m "shot_design: simulate subcommand with status file"
```

