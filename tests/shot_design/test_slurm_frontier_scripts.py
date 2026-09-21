"""House-style checks on the Frontier shot_design SLURM wrappers.

A wrapper is only ever exercised by submitting it, so the cheap mistakes -- the
wrong account, a Stellar path or QOS copied over with the body, logs written
into the repository -- are caught here instead of by a job that dies at minute
one of a six-hour allocation.
"""

import re
from pathlib import Path

import pytest

SLURM = Path(__file__).resolve().parents[2] / "scripts" / "slurm_frontier"
SCRIPTS = ["shot_design_census.sh", "shot_design_build.sh", "shot_design_encode.sh",
           "shot_design_simulate.sh", "shot_design_genc.sh"]


@pytest.mark.parametrize("name", SCRIPTS)
def test_frontier_shot_design_scripts_follow_house_style(name):
    text = (SLURM / name).read_text()
    assert "#SBATCH -A fus187" in text
    assert re.search(r"#SBATCH -p (batch|extended)", text)
    assert "_shot_design_common.sh" in text
    assert "/scratch/gpfs" not in text, "Stellar path leaked into a Frontier script"
    assert "gpu-stellar" not in text and "pppl" not in text
    assert "runs/slurm/%j" in text or "runs/slurm/%A_%a" in text


@pytest.mark.parametrize("name", SCRIPTS)
def test_wrappers_source_the_common_file_from_the_submit_dir(name):
    """`sbatch` executes a spool COPY of the script, so `dirname "$0"` is the spool
    directory, not the repo (job 5514036 died with `_shot_design_common.sh: No such
    file`). `$SLURM_SUBMIT_DIR` is where `sbatch` was invoked -- the repo root."""
    text = (SLURM / name).read_text()
    assert 'source "$(dirname "$0")/_shot_design_common.sh"' not in text
    assert ('source "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}'
            '/scripts/slurm_frontier/_shot_design_common.sh"') in text


def test_the_census_writes_the_parquet_name_every_consumer_defaults_to():
    """`corpus scan --out`, `select` and `coverage` all default to
    `<db_dir>/corpus_coverage.parquet`; a census under any other name is written and
    then never read."""
    text = (SLURM / "shot_design_census.sh").read_text()
    n = text.count('"$ROOT/db/corpus_coverage.parquet"')
    assert n == 2, "scan --out and summary must name the same file"
    assert "db/census.parquet" not in text


def test_the_gpu_sampler_measures_only_the_gcds_the_job_was_given():
    """rocm-smi ignores ROCR_VISIBLE_DEVICES and prints every card on the node, so
    without an explicit `-d` a one-GCD job on an exclusive node is averaged against idle
    siblings -- and the utilisation gate then judges that eighth."""
    text = (SLURM / "_gpu_sampler.sh").read_text()
    assert "ROCR_VISIBLE_DEVICES" in text
    assert "-d " in text


def test_build_wrapper_defaults_to_the_corpus_reader_without_encoding():
    # Frontier has no d3d_fusion_data raw layer: a build with the legacy reader skips every
    # shot ("no Ip signal on disk") and exits 1 (job 5515059). The corpus IS the raw layer,
    # and the IGNITE channel comes from shot_design_encode.sh.
    body = (SLURM / "shot_design_build.sh").read_text(encoding="utf-8")
    assert "${BUILD_ARGS:---reader corpus --no-encode}" in body


def test_build_wrapper_turns_the_llm_off_so_blurbs_wait_for_the_login_node():
    """Compute nodes see the agy binary on the shared filesystem but have no route to
    Google: build 5517788 spent its whole hour on 68 blurb calls that each waited ~50 s
    for an auth timeout. The build writes template blurbs; `blurb_frontier.sh` fills
    them in from the login node afterwards."""
    body = (SLURM / "shot_design_build.sh").read_text(encoding="utf-8")
    assert "export SHOT_DESIGN_LLM_PROVIDER=off" in body


def test_simulate_wrapper_does_not_set_debug_qos_so_the_demo_loop_can_submit():
    """Frontier's debug QOS caps `MaxSubmitJobsPU` at 1, so a wrapper that hardcodes
    `-q debug` breaks the moment a second job (the demo loop's next design, or the
    UI's second concurrent simulation) tries to submit while the first is still
    queued/running (job 5516504). The job is `-t 01:00:00` on `batch`, comfortably
    inside the debug QOS's own 2 h cap, so `-q debug` bought nothing but the
    one-job limit. A one-off run can still ask for it explicitly with
    `sbatch -q debug ...`."""
    text = (SLURM / "shot_design_simulate.sh").read_text()
    assert "#SBATCH -q debug" not in text
    assert "sbatch -q debug" in text, (
        "the wrapper should still document the debug-QOS opt-in for a one-off run"
    )


def test_common_file_skips_the_compute_node_settings_outside_a_job():
    # The login-node scripts (blurb backfill, demo) source the common file: it must not pull in
    # _frontier_settings.sh there, which needs SLURM_JOB_ID and SLURM_NODELIST (set -u dies).
    body = (SLURM / "_shot_design_common.sh").read_text(encoding="utf-8")
    assert 'if [[ -n "${SLURM_JOB_ID:-}" ]]; then' in body
    guarded = body.split('if [[ -n "${SLURM_JOB_ID:-}" ]]; then', 1)[1].split("fi", 1)[0]
    assert "_frontier_settings.sh" in guarded

