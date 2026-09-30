"""Measured script sizes and notes; scripts are read, never executed."""

from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "labeler"


def test_poi_pilot_matches_four_workers_and_memory_uses_the_full_run():
    text = (SCRIPTS / "ae_seg_poi.sbatch").read_text()
    assert "#SBATCH --mem=7500M" in text and "2945169" in text and "6,342,728K" in text
    pilot = next(line for line in text.splitlines() if "pilot:" in line)
    shots = pilot.split('SHOTS="')[1].split('"')[0].split()
    assert len(shots) >= 8
    assert "2942540" in text and "5,240,864K" in text
    assert "2942538" in text and "2700M" in text
    assert "two shots per worker" in text
    assert "workers × one worker's peak plus the parent" in text


def test_training_scheduler_note_names_the_jobs_that_actually_moved():
    text = (SCRIPTS / "ae_xpower_train.sbatch").read_text()
    for job in ("2942484_2", "2942491_0", "2942491_1"):
        assert job in text
    assert "serial" in text and "pppl-short-stellar" in text
    assert "2942492" in text and "2942516-2942518" in text


def test_extension_pilot_note_names_its_separate_output():
    text = (SCRIPTS / "ae_xpower_extend.sbatch").read_text()
    pilot = next(line for line in text.splitlines() if "pilot:" in line)
    assert "shards/pilot/" in pilot
