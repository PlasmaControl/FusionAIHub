### Task C1: Manifest-driven modality table and `shot_design model --pin`

**Files:**
- Modify: `src/tokamak_foundation_model/ignite/dynamics_config.py` (add `modalities_from_manifest`)
- Modify: `src/shot_design/shotdb/ignite.py` (`model_cfg`, `bundle_dir`, new `pin_bundle`, `load_codecs` manifest read, vocab check)
- Modify: `configs/shot_design/ignite_modalities.yaml` (`model:` block)
- Modify: `src/shot_design/cli.py` (`model` subcommand: `--pin`, `--check`)
- Test: `tests/shot_design/test_ignite_v4.py`, `tests/ignite/test_dynamics_config_manifest.py`

**Interfaces:**
- Produces: `dynamics_config.modalities_from_manifest(path: Path) -> tuple[ModalitySpec, ...]` reading `{"modalities": {name: {"family", "n_tok", "codebook_size"}}}` in file order.
- `shotdb.ignite.pin_bundle(paths, *, codec_tmpl: str, dynamics_src: Path, names: list[str], t0_start: float) -> Path` writes `<models_dir>/<local_name>/codecs/<m>/codec_best.pt` (real copies, `shutil.copy2`, symlinks resolved), `codecs/MANIFEST.json`, copies the dynamics file to `<local_name>/<dynamics_file>` and records sha256 of every file in the manifest under `"sha256"`.
- `shotdb.ignite.check_bundle(paths) -> list[str]` returns mismatches (empty = ok). `load_codecs` calls it and raises `CheckpointMissing` listing the mismatches when non-empty.
- `ignite_modalities.yaml` `model:` block:

```yaml
model:
  generation: v4
  local_name: IGNITE_v4
  codec_tmpl: /lustre/orion/fus187/proj-shared/models/ignite_codecs_v4/{m}/codec_best.pt
  dynamics_src: /lustre/orion/fus187/proj-shared/models/ignite_prod_v4/runs/mskfull/dynamics_best.pt
  dynamics_file: ignite_dynamics_prod_v4_mskfull_step3200.pt
  frame_codes_cache: /lustre/orion/fus187/proj-shared/models/ignite_prod_v4/frame_codes
  frame_tokens: 1209
  t0_start_s: 1.0
  window_ms: 250
  production_vocabs: {ece: 1000, bes: 1000, mhr: 1000, co2: 1000, mirnov: 1000, tangtv_lower: 1000, tangtv_upper: 1000, ts_core_density: 1000, ts_core_temp: 1000, ts_tangential_density: 1000, ts_tangential_temp: 1000, cer_ti: 1000, cer_rot: 1000, mse: 1000, filterscopes: 1000}
  families: {ece: spectro, bes: spectro, mhr: spectro, co2: spectro, mirnov: spectro, tangtv_lower: video, tangtv_upper: video, ts_core_density: slowts, ts_core_temp: slowts, ts_tangential_density: slowts, ts_tangential_temp: slowts, cer_ti: slowts, cer_rot: slowts, mse: slowts, filterscopes: fastts}
  n_tok: {ece: 192, bes: 192, mhr: 192, co2: 192, mirnov: 192, tangtv_lower: 108, tangtv_upper: 108, ts_core_density: 4, ts_core_temp: 4, ts_tangential_density: 4, ts_tangential_temp: 4, cer_ti: 4, cer_rot: 4, mse: 4, filterscopes: 5}
  # v2 bundle kept for reference / rollback: repo_id nc1/IGNITE rev d0cfe4f73d0287d5e12d724f8d77ed651fcfa47a
```
Keep the existing long header comment but rewrite the SCOPE paragraph for 15 modalities / 1209 tokens. Add a `mirnov:` entry to the `modalities:` map below (copy `mhr`'s structure; `staged.cols` listing the 29 mirnov channels from `data_loader.SIGNAL_CONFIGS`; `n_channels: 29`).

- [ ] **Step 1: Failing tests**

```python
# tests/ignite/test_dynamics_config_manifest.py
import json
from tokamak_foundation_model.ignite import dynamics_config as dc

def test_modalities_from_manifest_preserves_order_and_totals(tmp_path):
    m = {"modalities": {"ece": {"family": "spectro", "n_tok": 192, "codebook_size": 1000},
                        "mirnov": {"family": "spectro", "n_tok": 192, "codebook_size": 1000},
                        "filterscopes": {"family": "fastts", "n_tok": 5, "codebook_size": 1000}}}
    p = tmp_path / "MANIFEST.json"; p.write_text(json.dumps(m))
    mods = dc.modalities_from_manifest(p)
    assert [x.name for x in mods] == ["ece", "mirnov", "filterscopes"]
    assert sum(x.n_tok for x in mods) == 389
    assert mods[1] == dc.ModalitySpec("mirnov", "spectro", 192, 1000)
```

```python
# tests/shot_design/test_ignite_v4.py
import hashlib, json, torch, pytest
from shot_design.shotdb import ignite

def _fake_codec(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": {"d_model": 8}, "codec": {}}, path)

def test_pin_bundle_copies_resolves_symlinks_and_records_sha(paths, tmp_path):
    src = tmp_path / "codecs_v4"; real = tmp_path / "real"
    _fake_codec(real / "ece.pt"); (src / "ece").mkdir(parents=True)
    (src / "ece" / "codec_best.pt").symlink_to(real / "ece.pt")
    dyn = tmp_path / "dyn.pt"; torch.save({"step": 3200, "modalities": [("ece", "spectro", 192, 1000)], "model": {}}, dyn)
    out = ignite.pin_bundle(paths, codec_tmpl=str(src / "{m}" / "codec_best.pt"), dynamics_src=dyn, names=["ece"], t0_start=1.0)
    copied = out / "codecs" / "ece" / "codec_best.pt"
    assert copied.exists() and not copied.is_symlink()
    man = json.loads((out / "codecs" / "MANIFEST.json").read_text())
    assert man["modalities"]["ece"] == {"family": "spectro", "n_tok": 192, "codebook_size": 1000}
    assert man["sha256"]["codecs/ece/codec_best.pt"] == hashlib.sha256(copied.read_bytes()).hexdigest()
    assert man["t0_start_s"] == 1.0 and man["frame_tokens"] == 192
    assert ignite.check_bundle(paths) == []

def test_check_bundle_reports_a_changed_codec(paths, tmp_path):
    test_pin_bundle_copies_resolves_symlinks_and_records_sha(paths, tmp_path)
    p = ignite.bundle_dir(paths) / "codecs" / "ece" / "codec_best.pt"
    p.write_bytes(b"tampered")
    bad = ignite.check_bundle(paths)
    assert bad and "codecs/ece/codec_best.pt" in bad[0]
    with pytest.raises(ignite.CheckpointMissing):
        ignite.load_codecs(ignite.bundle_dir(paths))

def test_model_cfg_declares_fifteen_v4_modalities():
    cfg = ignite.model_cfg()
    assert cfg["generation"] == "v4"
    assert len(cfg["production_vocabs"]) == 15 and set(cfg["production_vocabs"].values()) == {1000}
    assert sum(cfg["n_tok"].values()) == cfg["frame_tokens"] == 1209
    assert cfg["t0_start_s"] == 1.0
```
The `pin_bundle` test uses `names=["ece"]` so `frame_tokens` in that manifest is 192; the function computes it from `n_tok` of the pinned names.

- [ ] **Step 2: Run to fail** → `AttributeError`.

- [ ] **Step 3: Implement**

`dynamics_config.modalities_from_manifest`: 6 lines, `json.loads`, tuple comprehension in dict order.

`shotdb/ignite.py`:
- `pin_bundle`: for each name, `src = Path(codec_tmpl.format(m=name)).resolve()`; copy to `bundle_dir(paths)/codecs/<name>/codec_best.pt`; copy `dynamics_src` to `bundle_dir(paths)/model_cfg()["dynamics_file"]`; write `MANIFEST.json`:
  `{"_meta": {"created", "generation", "codec_tmpl", "dynamics_src", "copy_mode": "shutil.copy2, symlinks resolved"}, "modalities": {name: {"family", "n_tok", "codebook_size"}}, "t0_start_s", "frame_tokens", "sha256": {relpath: hex}}` with families/n_tok/vocab from `model_cfg()["families"|"n_tok"|"production_vocabs"]`.
- `check_bundle`: recompute sha256 for each entry; return `[f"{rel}: expected {a[:12]} got {b[:12]}"]`.
- `load_codecs`: after reading `entries`, call `check_bundle` when `manifest` has `sha256`; raise `CheckpointMissing("pinned bundle changed on disk: ...")` if non-empty. Everything else unchanged (`entries[name]["family"]` is still what it reads).
- `download_bundle` stays for `generation: v2`; `cli.py` `model` gains `--pin` (calls `pin_bundle` with the yaml values and all 15 names) and `--check` (prints `check_bundle` result, exit 1 if non-empty); `--download` prints "not used for generation v4" when `generation != "v2"`.

- [ ] **Step 4: Run to pass**

```bash
PYTHONPATH=src .pixi/envs/frontier/bin/python -m pytest tests/ignite/test_dynamics_config_manifest.py tests/shot_design/test_ignite_v4.py tests/shot_design/test_ignite.py -q
```

- [ ] **Step 5: Pin the real bundle (login node, ~4 GB copy) and commit**

```bash
pixi run --frozen -e shot-design-frontier python -m shot_design model --pin && pixi run --frozen -e shot-design-frontier python -m shot_design model --check
ls -la /lustre/orion/fus187/proj-shared/nchen/shot_design/models/IGNITE_v4/codecs | head -20
git add configs/shot_design/ignite_modalities.yaml src/shot_design/shotdb/ignite.py src/shot_design/cli.py src/tokamak_foundation_model/ignite/dynamics_config.py tests/ignite/test_dynamics_config_manifest.py tests/shot_design/test_ignite_v4.py
git commit -m "ignite: v4 generation pinned by sha256 manifest; 15-modality table from manifest"
```

