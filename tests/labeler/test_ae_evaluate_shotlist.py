"""`ae_evaluate.py --shotlist`: named shot lists scored on the 10 ms frames, on a
synthetic dataset, reference table and two tiny checkpoints (CPU, no GPU)."""

from __future__ import annotations

import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from labeler.ae.model import AeSeldNet, AeSeldNetConfig
from labeler.config import Paths, sha256_of
from labeler.events import shotlists

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts" / "labeler"
T = 7820  # a 2 s record's 0.256 ms columns
CONFIG = AeSeldNetConfig(
    n_freq=12, pool_sizes=(12,), conv_channels=2, rnn_sizes=(4,), fnn_size=4
)
#: shot -> split of the synthetic npz; 14 has none
NPZ = {11: "train", 12: "valid", 13: "valid", 15: "valid"}
#: the reviewed table: 12 present 0-1 s; 13 present 200-600 ms, uncertain 600-800 ms;
#: 11 and 14 present 0-1 s; 15 is not in it
REVIEW = """shot,category,t_start,t_end,confidence,attrs
11,1,0,1000,,
11,0,1000,2000,,
12,1,0,1000,,
12,0,1000,2000,,
13,0,0,200,,
13,1,200,600,,
13,2,600,800,,
13,0,800,2000,,
14,1,0,1000,,
14,0,1000,2000,,
"""


@pytest.fixture(scope="module")
def ev():
    sys.path.insert(0, str(SCRIPTS))
    try:
        return importlib.import_module("ae_evaluate")
    finally:
        sys.path.remove(str(SCRIPTS))


@pytest.fixture
def tree(tmp_path, monkeypatch, ev):
    tables = tmp_path / "events"
    paths = Paths(
        root=tmp_path / "root", corpus=tmp_path / "corpus", text_root=tmp_path / "text",
        logs_jsonl=tmp_path / "logs.jsonl", label_tables=tables,
        config_dir=tmp_path / "config", raw_cache=tmp_path / "raw",
    )
    paths.shotlists.mkdir(parents=True)
    (paths.shotlists / "evalset.txt").write_text("11\n12\n13\n14\n15\n")
    (paths.shotlists / "twin.txt").write_text("13\n12\n")
    (paths.shotlists / "nothing.txt").write_text("14\n")
    review = tables / "alfven_eigenmode" / "review" / "labels.csv"
    review.parent.mkdir(parents=True)
    review.write_text(REVIEW)
    monkeypatch.setattr(Paths, "from_env", lambda: paths)

    rng = np.random.default_rng(3)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for shot, split in NPZ.items():
        annotated = np.zeros(T, bool)
        annotated[: T // 4] = True
        np.savez(
            dataset / f"{shot}_{split}.npz",
            spec=rng.normal(size=(4, CONFIG.n_freq, T)).astype(np.float16),
            annotated=annotated, active=annotated, freq_khz=np.full(T, np.nan),
            split=np.array(split),
        )
    models = tmp_path / "models"
    models.mkdir()
    for k, name in enumerate(("threeway_sce", "mask_sce")):
        torch.manual_seed(k)
        model = AeSeldNet(CONFIG)
        torch.save(
            {"config": CONFIG.as_dict(), "state_dict": model.state_dict(), "target": "threeway",
             "loss_config": {"loss": "sce"}, "epoch": 3, "val_loss": 0.5},
            models / f"ae_seldnet_{name}.pt",
        )
    out = tmp_path / "out" / "evaluation.json"
    # the corpus chain's checkpoint: the shipped test one, older than any cache
    checkpoint = models / "ae_seldnet_threeway_sce.pt"
    os.utime(checkpoint, (1e9, 1e9))
    monkeypatch.setattr(ev, "CORPUS_CHECKPOINT", checkpoint)
    monkeypatch.setattr(ev, "TRAINING_SHOTS", frozenset(NPZ))

    def run(*extra, workers=1):
        argv = ["--dataset-dir", str(dataset), "--models-dir", str(models), "--device", "cpu",
                "--out", str(out), "--workers", str(workers), *extra]
        assert ev.main(argv) == 0
        return json.loads(out.read_text())

    return {"paths": paths, "run": run, "out": out, "review": review, "dataset": dataset,
            "models": models, "checkpoint": checkpoint}


def test_a_list_scores_its_held_out_shots_and_names_every_other(tree, ev):
    record = tree["run"]("--shotlist", "evalset", "--shotlist", "twin,evalset")
    paths = tree["paths"]
    assert list(record["sets"]) == ["evalset", "twin"]
    assert record["sets"]["evalset"] == {
        "list": "evalset", "title": "evalset",
        "list_sha256": sha256_of(paths.shotlists / "evalset.txt"),
        "n_listed": 5, "n_scored": 2,
        "missing": {"no_input": [14], "not_in_cv": [], "no_reference": [15]},
        "trained_on": [11], "scoring": "heldout", "leaky": False,
    }
    assert record["leaky"] is False and record["include_trained"] is False
    assert record["population"] == {"npz": 4, "train": 1, "valid": 3}
    assert record["reference"]["path"] == str(tree["review"])
    assert record["reference"]["sha256"] == sha256_of(tree["review"])
    assert record["reference"]["n_shots"] == record["reference"]["n_covered"] == 4
    assert set(record["checkpoints"]) == {"threeway_sce", "mask_sce"}

    got = record["results"]["evalset"]["threeway_sce"]
    reviewed, annotated = got["reviewed"], got["annotated"]
    # 12's 200 frames, and 13's less the 20 uncertain ones; present 100 + 40
    assert (reviewed["n_shots"], reviewed["n_frames"]) == (2, 380)
    assert reviewed["truth_fraction"] == pytest.approx(140 / 380)
    assert set(reviewed["ci95"]) >= {"auroc", "auprc", "f1"} and "ci95" not in annotated
    assert (annotated["n_frames"], annotated["truth_fraction"]) == (400, 0.25)
    for key in ("auroc", "auprc", "precision", "recall", "f1", "best_f1"):
        assert key in reviewed
    # the same shots in another order: the same numbers, intervals included
    assert record["results"]["twin"] == record["results"]["evalset"]
    text = tree["out"].read_text()
    assert "NaN" not in text and "Infinity" not in text

    frames = np.load(tree["out"].with_name("evaluation.frames.npz"))
    assert frames["shots"].tolist() == [12, 13], "trained-on and unreferenced shots are not run"
    assert frames["threeway_sce_mean"].shape == (2, ev.N_FRAMES)
    # the frame means equal the headline scorer's on a whole-record pass of one shot
    blob = torch.load(tree["models"] / "ae_seldnet_threeway_sce.pt", weights_only=False)
    model = AeSeldNet(AeSeldNetConfig.from_dict(blob["config"]))
    model.load_state_dict(blob["state_dict"])
    with np.load(tree["dataset"] / "12_valid.npz") as z:
        x = torch.from_numpy(np.ascontiguousarray(z["spec"].transpose(0, 2, 1)))[None]
    with torch.no_grad():
        prob = torch.sigmoid(model.eval()(x.float())[0, :, 0]).double().numpy()
    np.testing.assert_allclose(frames["threeway_sce_mean"][0], ev.frame_mean(prob), atol=1e-6)
    np.testing.assert_allclose(
        frames["threeway_sce_share"][0], ev.frame_mean((prob >= 0.5).astype(float)), atol=1e-6
    )
    assert frames["annotated"][:, :50].all() and not frames["annotated"][:, 50:].any()


def test_include_trained_scores_the_trained_shots_and_marks_the_record_leaky(tree):
    record = tree["run"]("--shotlist", "evalset,nothing", "--include-trained")
    # two loader workers, then two scoring processes: the same numbers
    pooled = tree["run"]("--shotlist", "evalset,nothing", "--include-trained", workers=2)
    assert pooled["results"] == record["results"] and pooled["sets"] == record["sets"]
    env = record["sets"]["evalset"]
    assert (env["n_scored"], env["trained_on"], env["leaky"]) == (3, [11], True)
    assert record["leaky"] is True and record["include_trained"] is True
    assert record["results"]["evalset"]["threeway_sce"]["reviewed"]["n_shots"] == 3
    assert any("leaky" in text for text in record["cautions"]["evalset"])
    assert record["sets"]["nothing"]["n_scored"] == 0 and "nothing" not in record["results"]
    assert record["cautions"]["nothing"][0].startswith("no shot to score")


def test_a_run_with_nothing_to_score_still_writes_the_envelopes(tree):
    record = tree["run"]("--shotlist", "nothing")
    assert record["sets"]["nothing"]["missing"]["no_input"] == [14]
    assert record["results"] == {} and record["timing"]["shots"] == 0


def test_bad_arguments_and_unknown_lists_stop_with_a_message(tree, ev):
    with pytest.raises(SystemExit, match=r"shot list 'absent'.*known lists: .*evalset"):
        tree["run"]("--shotlist", "evalset,absent")
    for argv in (["--shotlist", " , "], ["--include-trained"]):
        with pytest.raises(SystemExit) as stop:
            ev.main(argv)
        assert stop.value.code == 2


def test_frame_scores_on_the_device_match_the_headline_scorer(ev):
    prob = np.random.default_rng(1).uniform(size=(3, T))
    mean, share = ev.frame_scores(torch.from_numpy(prob))
    for row in range(3):
        np.testing.assert_allclose(mean[row].numpy(), ev.frame_mean(prob[row]), atol=1e-12)
        np.testing.assert_allclose(
            share[row].numpy(), ev.frame_mean((prob[row] >= 0.5).astype(float)), atol=1e-12
        )
    assert ev.list_names(["a,b", " c ,a", ""]) == ["a", "b", "c"]


def corpus_npz(path, end_ms, phase, run_on=None):
    """A deployed-chain cache as ae_gold_extend.py writes it, columns to past `end_ms`."""
    t = np.arange(int((end_ms + 10) / 0.256)) * 0.256 - 0.768
    extra = {"run_on": np.array(run_on)} if run_on else {}
    np.savez(
        path, t_ms=t, prob=(0.5 + 0.45 * np.sin(t / 37.0 + phase)).astype(np.float32),
        plasma_end_ms=np.float64(end_ms), plasma_end_how=np.array("test"),
        input_end_ms=np.float64(t[-1]), fs_hz=np.float64(5e5), **extra,
    )


def test_the_corpus_chain_scores_the_deployed_caches_beside_the_dataset_chain(tree, ev):
    # 16 is uncertain inside its window and present only after it ends
    tree["review"].write_text(tree["review"].read_text() + "16,2,0,1000,,\n16,1,1500,2000,,\n")
    (tree["paths"].shotlists / "late.txt").write_text("16\n")
    (tree["paths"].shotlists / "gap.txt").write_text("17\n")  # no cache anywhere
    before = tree["run"]("--shotlist", "evalset")
    first, second = tree["out"].parent / "gold_work", tree["out"].parent / "silver_work"
    first.mkdir()
    second.mkdir()
    corpus_npz(first / "12.npz", 3000.0, 0.0)
    corpus_npz(second / "12.npz", 3000.0, 1.0, run_on="chain gpu")  # the first dir wins
    corpus_npz(second / "13.npz", 705.0, 2.0, run_on="chain gpu")  # frames 0-69 assessed
    corpus_npz(first / "14.npz", 1500.0, 3.0)  # no dataset npz, but a corpus one
    corpus_npz(second / "11.npz", 3000.0, 4.0)  # train split
    corpus_npz(second / "16.npz", 1000.0, 5.0)
    corpus_npz(second / "15.npz", 3000.0, 6.0)  # not in the review table
    record = tree["run"](
        "--shotlist", "evalset,late,gap", "--chain", "corpus",
        "--corpus-work", str(first), "--corpus-work", str(second), workers=2,
    )
    assert {k: v for k, v in record.items() if k != "corpus"} == before, "the dataset chain's numbers stay"
    corpus = record["corpus"]
    assert corpus["sets"]["evalset"] == {
        **before["sets"]["evalset"], "n_scored": 3,
        "missing": {"no_input": [], "not_in_cv": [], "no_reference": [15]},
    }
    assert corpus["sets"]["late"]["missing"]["no_reference"] == [16]
    assert corpus["unscorable"] == {
        "evalset": {"15": "not in the review table",
                    "11": "train split: the checkpoint was fitted on it"},
        "late": {"16": "its reviewed frames all lie outside the chain's window"},
        "gap": {"17": "no corpus-chain npz"},
    }
    assert corpus["cautions"]["gap"][0].startswith("no shot to score")
    got = corpus["results"]["evalset"]["reviewed"]
    # 12's 200 frames, 13's 60 inside its window less the uncertain ones, 14's 150
    assert (got["n_shots"], got["n_frames"]) == (3, 410)
    assert got["truth_fraction"] == pytest.approx(240 / 410)
    assert set(got["ci95"]) >= {"auroc", "auprc", "f1"} and "late" not in corpus["results"]
    assert corpus["duplicates"] == {"12": [str(first), str(second)]}
    assert corpus["run_on"] == {"chain gpu": 1, "not recorded": 5}
    assert [d["listed_used"] for d in corpus["work_dirs"]] == [2, 4]
    assert all(d["checkpoint"]["same_checkpoint"] for d in corpus["work_dirs"])
    assert corpus["checkpoint"]["sha256"] == sha256_of(tree["checkpoint"])
    assert corpus["checkpoint"]["same_as_dataset_chain_shipped"] is True
    assert (corpus["trained_on_from"]["train"], corpus["trained_on_from"]["valid"]) == (1, 3)
    assert corpus["reference"]["same_as_dataset_chain"] is True
    # no review history beside the reference: nothing is left out, and every list says so
    assert corpus["prefilled"]["checked"] is False and corpus["prefilled"]["shots"] == {}
    assert all(any("unchecked" in c for c in said) for said in corpus["cautions"].values())

    frames = np.load(tree["out"].with_name("evaluation.corpus_frames.npz"))
    assert frames["shots"].tolist() == [11, 12, 13, 14, 15, 16]
    assert frames["assessed"].sum(axis=1).tolist() == [200, 200, 70, 150, 200, 100]
    sys.path.insert(0, str(SCRIPTS))
    try:
        extend = importlib.import_module("ae_gold_extend")
    finally:
        sys.path.remove(str(SCRIPTS))
    with np.load(first / "12.npz") as z:
        share, mean, _ = extend.frames(z["t_ms"], z["prob"].astype(np.float64), ev.N_FRAMES)
    np.testing.assert_allclose(frames["mean"][1], mean, atol=1e-12)
    np.testing.assert_allclose(frames["share"][1], share, atol=1e-12)

    # the shots both chains scored (12 and 13), on the frames the corpus chain assesses
    agree = corpus["agreement"]
    assert (agree["n_shots"], agree["n_frames"]) == (2, 270)
    assert agree["reviewed"]["dataset"]["n_frames"] == agree["reviewed"]["corpus"]["n_frames"] == 260
    assert -1 <= agree["pearson_r"] <= 1 and 0 <= agree["call_agreement"] <= 1
    assert set(agree["per_shot"]) == {"12", "13"}
    assert "NaN" not in tree["out"].read_text()


def history_line(shot, source, prefilled="unset", **extra):
    entry = {"shot": shot, "reviewer": "owner", "saved_at": f"2026-10-15T0{shot % 10}:00:00+00:00",
             "window": [0, 2000], "intervals": [], "source": source, **extra}
    if prefilled != "unset":  # old clients wrote no prefilled key
        entry["prefilled"] = prefilled
    return json.dumps(entry) + "\n"


def write_sidecar(table, meta):
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("shot,category,t_start,t_end,confidence\n12,1,0,500,0.9\n")
    table.with_name(table.stem + ".meta.json").write_text(json.dumps(meta))


def test_truth_prefilled_from_this_model_is_left_out_of_both_chains(tree, ev):
    event = tree["review"].parent.parent
    tree["review"].write_text(tree["review"].read_text() + "16,1,0,1000,,\n16,0,1000,2000,,\n")
    (tree["paths"].shotlists / "late.txt").write_text("16\n")
    sha = sha256_of(tree["checkpoint"])
    gold = event / "extend_ae_seldnet" / "gold_shots.csv"
    write_sidecar(gold, {"model": {"checkpoint_sha256": sha}})
    write_sidecar(event / "format" / "alfven_eigenmode_format_2026_v1.csv", {"producer": "hand"})
    # a layer derived from the model's table, kept elsewhere
    write_sidecar(tree["paths"].root / "suggestions" / "ae_vote" / "x.csv",
                  {"inputs": ["alfven_eigenmode/extend_ae_seldnet/gold_shots"]})
    model = "extend:alfven_eigenmode/extend_ae_seldnet/gold_shots"
    (event / "review" / "history.jsonl").write_text("".join((
        history_line(12, model, True, source_sha256=sha256_of(gold)),
        history_line(12, "alfven_eigenmode_format_2026_v1.csv"),  # opened on 12's saved label
        history_line(13, "alfven_eigenmode_format_2026_v1.csv"),  # the hand table: independent
        history_line(14, "suggest:ae_vote/x", True),  # derived from the model's table
        history_line(16, model, True, source_sha256=sha256_of(gold)),
        # what the AE page writes when the saved shot is reopened after a reload and saved
        # again: prefilled false, though it opened on the model-made label
        history_line(16, model, False, source_sha256=sha256_of(gold)),
        history_line(11, "extend:alfven_eigenmode/extend_gone/x", True),  # unknown producer
    )))
    dataset = tree["run"]("--shotlist", "evalset,late")
    model_says, unknown = "truth prefilled from this model", (
        "truth prefilled from a table of unknown producer (it may be this model)")
    train = "train split: the checkpoint was fitted on it"
    assert sorted(dataset["prefilled"]["shots"]) == ["11", "12", "14", "16"]
    twelve, sixteen = dataset["prefilled"]["shots"]["12"], dataset["prefilled"]["shots"]["16"]
    assert (twelve["saves"], twelve["later_saves"], twelve["table_now"]) == (2, 1, True)
    assert (sixteen["saves"], sixteen["later_saves"]) == (2, 1), "a re-save keeps the anchor"
    assert dataset["prefilled"]["prefilled_false_clears"] is False
    assert twelve["how"] == "the table's sidecar names the checkpoint"
    assert dataset["prefilled"]["shots"]["14"]["how"] == "a layer of this model or derived from one"
    assert dataset["prefilled"]["model_layers"] == ["extend_ae_seldnet", "extend_ae_seldnet/gold_shots"]
    assert dataset["sets"]["evalset"]["n_scored"] == 1
    assert dataset["sets"]["evalset"]["missing"]["no_reference"] == [12, 15]
    assert dataset["unscorable"] == {
        "evalset": {"14": f"no dataset npz; and {model_says}", "11": f"{train}; and {unknown}",
                    "12": model_says, "15": "not in the review table"},
        "late": {"16": f"no dataset npz; and {model_says}"},
    }
    assert any("prefilled from this model" in c and "1 for that alone" in c
               and c.endswith(": 11, 12, 14") for c in dataset["cautions"]["evalset"])

    work = tree["out"].parent / "work"
    work.mkdir(parents=True)
    for shot in (11, 12, 13, 14, 16):
        corpus_npz(work / f"{shot}.npz", 3000.0, float(shot))
    args = ("--shotlist", "evalset,late", "--chain", "corpus", "--corpus-work", str(work))
    corpus = tree["run"](*args)["corpus"]
    assert corpus["prefilled"]["same_as_dataset_chain"] is True
    assert corpus["unscorable"] == {
        "evalset": {"15": "no corpus-chain npz", "11": f"{train}; and {unknown}",
                    "12": model_says, "14": model_says},
        "late": {"16": model_says},
    }
    assert corpus["results"]["evalset"]["reviewed"]["n_shots"] == 1  # 13 alone
    assert "late" not in corpus["results"]
    assert corpus["agreement"]["n_shots"] == 1  # 13: no prefilled shot in the reviewed check

    # only a page that carries exposure across saves (detachment's) clears the anchor
    _, tracked = ev.model_prefills(tree["paths"], tree["review"], {sha}, event="detachment")
    assert tracked["n_shots"] == 3 and tracked["prefilled_false_clears"] is True
    # the owner's call, by hand: 16 was re-labelled from scratch, so it counts again
    cleared = tree["run"](*args, "--independent-truth", "16,13")["corpus"]
    assert cleared["results"]["late"]["reviewed"]["n_frames"] == 200
    assert list(cleared["prefilled"]["cleared_by_hand"]) == ["16"]
    assert cleared["prefilled"]["cleared_not_anchored"] == [13]
    # a derived layer that names its input only by the extend CSV's sha256
    side = tree["paths"].root / "suggestions" / "ae_vote" / "x.meta.json"
    side.write_text(json.dumps({"input_sha256": sha256_of(gold)}))
    shots, summary = ev.model_prefills(tree["paths"], tree["review"], {sha})
    assert shots[14]["how"] == "a layer of this model or derived from one"
    assert summary["layer_sha256"] == {"extend_ae_seldnet/gold_shots": sha256_of(gold)}


def test_the_corpus_chain_fails_closed(tree, ev, monkeypatch):
    work = tree["out"].parent / "work"
    work.mkdir(parents=True)
    for shot in (11, 12, 13, 14):
        corpus_npz(work / f"{shot}.npz", 3000.0, float(shot))
    args = ("--shotlist", "evalset", "--chain", "corpus", "--corpus-work", str(work))
    empty = tree["dataset"].parent / "no_dataset"
    with pytest.raises(SystemExit, match="no <shot>_train.npz"):
        ev.main(["--dataset-dir", str(empty), "--out", str(tree["out"]), *args])
    # a card training shot with no npz in the dataset dir counts as trained on
    monkeypatch.setattr(ev, "TRAINING_SHOTS", frozenset({*NPZ, 14}))
    corpus = tree["run"](*args)["corpus"]
    assert corpus["sets"]["evalset"]["trained_on"] == [11, 14]
    assert corpus["trained_on_from"]["card_training_shots_without_npz"] == [14]
    # a checkpoint changed after the caches were made: they are not its output
    os.utime(tree["checkpoint"])
    with pytest.raises(SystemExit, match="caches not shown to come from"):
        tree["run"](*args)


def test_the_two_blocks_keep_each_other_and_score_one_reference(tree):
    work = tree["out"].parent / "work"
    work.mkdir(parents=True)
    corpus_npz(work / "12.npz", 3000.0, 0.0)
    corpus_args = ("--shotlist", "evalset", "--chain", "corpus", "--corpus-work", str(work))
    tree["run"]("--shotlist", "evalset")
    corpus = tree["run"](*corpus_args)["corpus"]
    rerun = tree["run"]("--shotlist", "evalset")
    kept = rerun["corpus"].pop("kept_by_dataset_run")
    assert rerun["corpus"] == corpus and kept["same_reference"] is True
    tree["review"].write_text(tree["review"].read_text() + "17,1,0,1000,,\n")
    with pytest.raises(SystemExit, match="rerun the dataset chain on it first"):
        tree["run"](*corpus_args)
    # a dataset run on the changed reference drops the old block and says so
    dropped = tree["run"]("--shotlist", "evalset")
    assert "corpus" not in dropped
    assert dropped["corpus_dropped"]["reference_sha256"] == corpus["reference"]["sha256"]
    assert "corpus_dropped" not in tree["run"](*corpus_args)  # the corpus run restores it


def test_the_corpus_chain_flags_go_together(tree, ev):
    for argv in (["--chain", "corpus"], ["--shotlist", "evalset", "--corpus-work", "x"]):
        with pytest.raises(SystemExit) as stop:
            ev.main(argv)
        assert stop.value.code == 2
    assert ev.kappa(np.array([True, False]), np.array([True, False])) == 1.0


def test_the_sbatch_is_one_gpu_job_over_the_ae_evaluate_lists(ev):
    path = SCRIPTS / "ae_evaluate.sbatch"
    subprocess.run(["bash", "-n", str(path)], check=True)
    text = path.read_text()
    for flag in ("--partition=gpu", "--qos=gpu-stellar", "--gres=gpu:1", "--cpus-per-task=8"):
        assert f"#SBATCH {flag}\n" in text, flag
    assert "--array" not in text
    assert "utilization.gpu" in text and "-l 2 >" in text and "labeler.jobstats --job-id" in text
    default = re.search(r'SHOTLIST="\$\{SHOTLIST:-([a-z0-9_,]+)\}"', text)[1].split(",")
    config = Paths(config_dir=REPO / "configs" / "labeler")
    assert sorted(default) == sorted(shotlists.evaluate(config, ev.EVENT))
