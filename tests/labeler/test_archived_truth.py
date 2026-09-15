"""Ground truth for one shot, and scoring a published label against it.

The tearing archive is the only ground truth labelmaker has, and it carries no
time axis of its own - its rows are placed on our grid by the same match
`validate` uses to score a pool. These helpers expose that per shot, so
`analyze` can say how a label did on the shot in front of you.
"""
import numpy as np
import pytest

from labelmaker import validate
from labelmaker.config import Paths

N = 6


class _FakeBuilt:
    def __init__(self):
        self.t = 0.025 * np.arange(8, dtype=np.float64)
        self.scalars = np.zeros((8, 9))
        self.profiles = np.zeros((8, 33, 0))
        self.valid = np.ones(8, bool)
        self.missing = ()
        self.resolvers = {}
        self.invalid_reasons = {}


def _fake_match(monkeypatch, tm_label, betan=None, index=(0, 1, 2, 4, 5, 6)):
    """Six archived rows landing on our timesteps `index`."""
    index = np.asarray(index)
    y = np.stack([
        np.asarray(betan if betan is not None else np.full(N, 2.0), dtype=float),
        np.asarray(tm_label, dtype=float),
    ], axis=1)
    got = {"x0": np.zeros((N, 9)), "x1": np.zeros((N, 33, 0)), "y": y,
           "rows": np.arange(N)}
    info = {"index": index, "passed": True, "n_archived_rows": N, "n_matched": N,
            "n_unique_matched": N, "median_distance": 0.0, "fail_reason": None}

    def fake(shot, spec, paths, archive):
        if shot != 190000:
            return validate._ShotMatch(skip_reason="no archived rows")
        return validate._ShotMatch(got=got, built=_FakeBuilt(), info=info, features={})

    monkeypatch.setattr(validate, "_matched_shot", fake)


def test_archived_truth_puts_the_archive_rows_on_our_time_axis(monkeypatch):
    _fake_match(monkeypatch, tm_label=[0, 0, 0, 1, 1, 0], betan=[1.0, 1.5, 2.0, 2.5, 3.0, 2.0])
    got = validate.archived_truth(190000, Paths.from_env())
    assert got["available"] is True
    np.testing.assert_array_equal(got["index"], [0, 1, 2, 4, 5, 6])
    np.testing.assert_allclose(got["t"], [0.0, 0.025, 0.05, 0.1, 0.125, 0.15])
    np.testing.assert_array_equal(got["tm_label"], [False, False, False, True, True, False])
    np.testing.assert_allclose(got["betan"], [1.0, 1.5, 2.0, 2.5, 3.0, 2.0])
    # the first row the archive calls a mode
    assert got["onset_s"] == pytest.approx(0.1)
    assert got["n_rows"] == 6


def test_archived_truth_reports_no_onset_when_the_label_never_turns_on(monkeypatch):
    _fake_match(monkeypatch, tm_label=[0] * N)
    got = validate.archived_truth(190000, Paths.from_env())
    assert got["available"] is True and got["onset_s"] is None


def test_archived_truth_says_why_a_shot_has_none(monkeypatch):
    _fake_match(monkeypatch, tm_label=[0] * N)
    got = validate.archived_truth(199597, Paths.from_env())
    assert got["available"] is False and got["reason"] == "no archived rows"
    assert "t" not in got


def test_a_binary_label_is_scored_against_the_archived_label_with_its_lead_time(monkeypatch):
    _fake_match(monkeypatch, tm_label=[0, 0, 0, 1, 1, 0])
    truth = validate.archived_truth(190000, Paths.from_env())
    # a label that crosses 0.5 two steps before the archived onset at 0.100 s
    y = np.array([0.1, 0.2, 0.6, 0.9, 0.8, 0.7, 0.2, 0.1])
    got = validate.score_against_truth(
        "d3d_tearing_onset_cnn1d/tm_prob", y, np.ones(8, bool), truth, threshold=0.5
    )
    assert got["scored"] is True and got["kind"] == "archived label"
    assert got["n"] == 6 and got["n_positive"] == 2
    assert got["auroc"] == pytest.approx(1.0)      # both positives rank above every negative
    # at 0.5 the label also flags the 0.6 row, which the archive calls quiet
    assert got["recall"] == pytest.approx(1.0) and got["precision"] == pytest.approx(2 / 3)
    assert got["onset_s"] == pytest.approx(0.1)
    assert got["first_above_threshold"] == pytest.approx(0.05)
    assert got["lead_time_s"] == pytest.approx(0.05)


def test_a_regression_label_is_scored_against_its_archived_column(monkeypatch):
    _fake_match(monkeypatch, tm_label=[0] * N, betan=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    truth = validate.archived_truth(190000, Paths.from_env())
    y = np.full(8, 1.5)
    got = validate.score_against_truth(
        "d3d_tearing_onset_cnn1d/betan", y, np.ones(8, bool), truth, threshold=None
    )
    assert got["scored"] is True and got["kind"] == "archived column"
    assert got["rmse"] == pytest.approx(0.5) and got["bias"] == pytest.approx(0.5)
    assert got["n"] == 6 and "auroc" not in got


def test_a_survival_label_is_scored_on_pre_onset_rows_within_its_horizon(monkeypatch):
    # onset at 0.100 s; rows at 0.000, 0.025, 0.050 are pre-onset. The
    # horizon is overridden to 50 ms so the boundary falls inside this
    # six-row fixture; a caller passes the label's own horizon.
    _fake_match(monkeypatch, tm_label=[0, 0, 0, 1, 1, 0])
    truth = validate.archived_truth(190000, Paths.from_env())
    y = np.array([0.05, 0.1, 0.3, 0.9, 0.9, 0.9, 0.1, 0.1])
    got = validate.score_against_truth(
        "d3d_tearing_time_to_event_dsm/tm_risk_250ms", y, np.ones(8, bool), truth,
        threshold=0.2, horizon_s=0.05,
    )
    assert got["scored"] is True and got["kind"] == "onset within 0.05 s"
    # rows at 0.050 (onset - t = 0.050 <= 0.05) is positive; 0.000 and 0.025 are not
    assert got["n"] == 3 and got["n_positive"] == 1
    assert got["auroc"] == pytest.approx(1.0)
    assert got["lead_time_s"] == pytest.approx(0.05)


def test_a_label_the_archive_cannot_serve_is_left_unscored(monkeypatch):
    _fake_match(monkeypatch, tm_label=[0] * N)
    truth = validate.archived_truth(190000, Paths.from_env())
    got = validate.score_against_truth("fake_model/score", np.zeros(8), np.ones(8, bool),
                                       truth, threshold=0.5)
    assert got["scored"] is False and "fake_model/score" in got["reason"]


def test_scoring_an_unavailable_truth_repeats_its_reason(monkeypatch):
    _fake_match(monkeypatch, tm_label=[0] * N)
    truth = validate.archived_truth(199597, Paths.from_env())
    got = validate.score_against_truth("d3d_tearing_onset_cnn1d/tm_prob", np.zeros(8),
                                       np.ones(8, bool), truth, threshold=0.5)
    assert got["scored"] is False and got["reason"] == "no archived rows"


def test_only_rows_labelmaker_would_publish_are_scored(monkeypatch):
    """A row the validity mask rejects is not a prediction labelmaker stands
    behind, so it must not count for or against the model."""
    _fake_match(monkeypatch, tm_label=[0, 0, 0, 1, 1, 0])
    truth = validate.archived_truth(190000, Paths.from_env())
    y = np.array([0.1, 0.2, 0.6, 0.9, 0.8, 0.7, 0.2, 0.1])
    valid = np.ones(8, bool)
    valid[4] = False                                   # drops one positive row
    got = validate.score_against_truth(
        "d3d_tearing_onset_cnn1d/tm_prob", y, valid, truth, threshold=0.5
    )
    assert got["n"] == 5 and got["n_positive"] == 1
    assert got["n_invalid_dropped"] == 1


def test_reaching_for_the_truth_never_raises(monkeypatch):
    """The archive may be unmounted, or hold rows this shot's match cannot
    place. Either way the caller keeps its analysis and gets a reason."""
    def boom(shot, spec, paths, archive):
        raise OSError("archive not mounted")

    monkeypatch.setattr(validate, "_matched_shot", boom)
    got = validate.archived_truth(190000, Paths.from_env())
    assert got["available"] is False and "not mounted" in got["reason"]


def test_survival_final_alarm_warns_before_onset(monkeypatch):
    _fake_match(monkeypatch, tm_label=[0, 0, 0, 1, 1, 0])
    truth = validate.archived_truth(190000, Paths.from_env())
    got = validate.score_against_truth(
        'd3d_tearing_time_to_event_dsm/tm_risk_1s',
        np.array([0, 0, .8, 0, 0, 0, 0, 0]), np.ones(8, bool), truth, threshold=.5)
    assert got['verdict'] == 'TP' and got['alarm'] is True
    assert got['warning_time_s'] == pytest.approx(.05)
    assert got['jumps'] == got['n_excursions'] == 0


def test_quiet_spike_is_any_row_but_not_final_alarm(monkeypatch):
    _fake_match(monkeypatch, tm_label=[0]*N)
    got = validate.score_against_truth(
        'd3d_tearing_onset_cnn1d/tm_prob', np.array([0, 1, 0, 0, 0, 0, 0, 0]),
        np.ones(8, bool), validate.archived_truth(190000, Paths.from_env()), threshold=.5)
    assert got['verdict'] == 'TN' and got['alarm'] is False
    assert got['any_row_call'] is True


@pytest.mark.parametrize('last_horizon', [1., 2.])
def test_pooled_rows_and_alarm_quality(wired, monkeypatch, last_horizon):
    from labelmaker.labels.schema import LabelSpec
    from labelmaker.labels.store import write_labels
    from labelmaker.models.base import Decoded

    slug = 'd3d_tearing_time_to_event_dsm'
    paths = Paths(root=wired['root'], corpus=wired['corpus'])
    _fake_match(monkeypatch, tm_label=[0, 0, 0, 1, 1, 0])
    original = validate._matched_shot

    def match(shot, spec, paths, archive):
        m = original(190000, spec, paths, archive)
        m.got['y'][:, 1] = [0, 0, 0, 1, 1, 0] if shot == 190000 else 0
        return m

    monkeypatch.setattr(validate, '_matched_shot', match)
    monkeypatch.setitem(validate.ARCHIVE_TRUTH, f'{slug}/tm_risk_1s',
                        {'kind': 'onset_within', 'horizon_s': last_horizon})
    names = ['tm_risk_250ms', 'tm_risk_500ms', 'tm_risk_1s', 'tm_time_p50']
    specs = tuple(LabelSpec(name=n, task='regression' if n.endswith('_p50') else 'binary',
                           activation='none', units='',
                           classes=(), slug=slug, card_id='test/model', time_step_ms=25,
                           ensemble_n=1, artifact_sha256='abc') for n in names)
    for shot in [190000, 190001]:
        y = np.array([.1, .8, .8, .2, .1, np.nan, .1, .9])
        valid = np.ones(8, bool)
        valid[0] = False
        write_labels(paths.labels_file(shot), shot, _FakeBuilt().t,
                     {n: Decoded(mean=y, lo=y, hi=y) for n in names}, specs, valid,
                     run_id='test', features_sha256='abc')
    pooled = validate._pooled_onset_rows(slug, [190000, 190001, 99], paths,
                                        archive=wired['archive'], timeout_s=None)
    regression = pooled["labels"]["tm_time_p50"]
    np.testing.assert_array_equal(regression["shot"], [190000] * 2)
    np.testing.assert_allclose(regression["truth"], [75.0, 50.0])
    assert regression["row_set"] == "pre-onset valid rows of aligned tearing shots only"
    r = pooled['labels']['tm_risk_1s']
    np.testing.assert_array_equal(r['shot'], [190000]*2 + [190001]*4)
    np.testing.assert_allclose(r['t'], [.025, .05, .025, .05, .1, .15])
    np.testing.assert_allclose(r['t_end'], [.15]*6)
    assert np.isnan(r['onset_s'][2:]).all()
    assert pooled['shots_used'] == [190000, 190001]
    assert pooled['shots_with_onset'] == [190000]
    assert 'labels' in pooled['skipped']['99']
    report = validate.alarm_quality(slug, [190000, 190001, 99], paths,
                                    archive=wired['archive'], thresholds=(.5,))
    assert 'tm_time_p50' not in report['labels']
    label = report['labels']['tm_risk_1s']['subsets']['all']
    assert label['n_quiet'] == label['n_tearing'] == 1
    rates = label['thresholds']['0.5']
    assert rates['final_label']['fpr'] == rates['final_label']['fnr'] == 0
    assert rates['any_row']['fpr'] == 1 and rates['any_row']['fnr'] == 0
    assert rates['warning_time_s']['median'] == pytest.approx(.075)
    assert label['auroc'] is not None and label['ipcw_auc'] is None  # no T > 1
    assert report['horizon_integrated']['all']['0.5']['fpr'] == last_horizon - .25


def _adapter_with_training_shots(monkeypatch, shots):
    """The `wired` fake adapter, but naming some shots as the model's own.

    Every survival number on the pool is part in-sample - 214 of the 500 pool
    shots are training shots - so the split has to come from the adapter the
    report is about, not from a hard-coded list in `validate`.
    """
    from dataclasses import replace

    from labelmaker.models import registry

    from .test_run import _fake_adapter

    adapter = replace(_fake_adapter(), training_shots=frozenset(shots))
    monkeypatch.setattr(registry, "load_adapter", lambda slug: adapter)
    return adapter


def _write_pooled_labels(paths, slug, shots, names):
    """One published trace per shot: eight steps, one invalid and one NaN."""
    from labelmaker.labels.schema import LabelSpec
    from labelmaker.labels.store import write_labels
    from labelmaker.models.base import Decoded

    specs = tuple(LabelSpec(name=n, task="binary", activation="none", units="",
                            classes=(), slug=slug, card_id="test/model", time_step_ms=25,
                            ensemble_n=1, artifact_sha256="abc") for n in names)
    y = np.array([.1, .8, .8, .2, .1, np.nan, .1, .9])
    valid = np.ones(8, bool)
    valid[0] = False
    for shot in shots:
        write_labels(paths.labels_file(shot), shot, _FakeBuilt().t,
                     {n: Decoded(mean=y, lo=y, hi=y) for n in names}, specs, valid,
                     run_id="test", features_sha256="abc")


def test_pooled_rows_and_alarm_quality_split_held_out_from_in_training(wired, monkeypatch):
    """Two training shots of four: every metric is reported three ways."""
    slug = "d3d_tearing_time_to_event_dsm"
    paths = Paths(root=wired["root"], corpus=wired["corpus"])
    shots = [190000, 190001, 190002, 190003]
    tearing = {190000, 190002}
    _fake_match(monkeypatch, tm_label=[0, 0, 0, 1, 1, 0])
    original = validate._matched_shot

    def match(shot, spec, paths, archive):
        m = original(190000, spec, paths, archive)
        m.got["y"][:, 1] = [0, 0, 0, 1, 1, 0] if shot in tearing else 0
        return m

    monkeypatch.setattr(validate, "_matched_shot", match)
    _adapter_with_training_shots(monkeypatch, {190000, 190001})
    monkeypatch.setitem(validate.ARCHIVE_TRUTH, f"{slug}/tm_risk_1s",
                        {"kind": "onset_within", "horizon_s": 1.0})
    names = ["tm_risk_250ms", "tm_risk_500ms", "tm_risk_1s"]
    _write_pooled_labels(paths, slug, shots, names)

    pooled = validate._pooled_onset_rows(slug, shots, paths,
                                         archive=wired["archive"], timeout_s=None)
    rows = pooled["labels"]["tm_risk_1s"]
    np.testing.assert_array_equal(
        rows["shot"], [190000] * 2 + [190001] * 4 + [190002] * 2 + [190003] * 4)
    np.testing.assert_array_equal(rows["in_training"], [True] * 6 + [False] * 6)
    assert pooled["shots_in_training"] == [190000, 190001]
    assert rows["shots_in_training"] == [190000, 190001]

    report = validate.alarm_quality(slug, shots, paths, archive=wired["archive"],
                                    thresholds=(.5,))
    assert report["shots_in_training"] == [190000, 190001]
    label = report["labels"]["tm_risk_1s"]
    assert set(label["subsets"]) == {"all", "held_out", "in_training"}
    assert {name: (s["n_shots"], s["n_quiet"], s["n_tearing"], s["n_rows"], s["n_positive"])
            for name, s in label["subsets"].items()} == {
        "all": (4, 2, 2, 12, 4), "held_out": (2, 1, 1, 6, 2), "in_training": (2, 1, 1, 6, 2)}
    for name in ("all", "held_out", "in_training"):
        rates = label["subsets"][name]["thresholds"]["0.5"]
        assert rates["final_label"]["fpr"] == rates["final_label"]["fnr"] == 0
        assert rates["any_row"]["fpr"] == 1 and rates["any_row"]["fnr"] == 0
        assert rates["warning_time_s"]["median"] == pytest.approx(.075)
        # The two halves are mirror images here, so the ranking is the same.
        assert label["subsets"][name]["auroc"] == pytest.approx(.75)
        assert report["horizon_integrated"][name]["0.5"]["fpr"] == .75


@pytest.mark.parametrize("published", [("risk_a", "risk_b"), ("risk_b",)])
def test_horizon_integral_follows_the_rules_not_one_slug(monkeypatch, published):
    """Any slug with two published onset_within labels gets the integral.

    A slug that publishes only one of its horizons leaves the integral out
    rather than raising - the report is about what is on disk.
    """
    rows = {name: {"shot": np.ones(2, int), "t": np.array([.0, .1]),
                   "y": np.array([.9, .9]), "truth": np.ones(2, bool),
                   "onset_s": np.full(2, .5), "t_end": np.full(2, .5),
                   "in_training": np.zeros(2, bool), "shots_used": [1],
                   "shots_with_onset": [1], "shots_in_training": [], "skipped": {},
                   "row_set": "pre-onset valid rows of every aligned shot"}
            for name in published}
    for name, horizon in (("risk_a", .25), ("risk_b", 1.)):
        monkeypatch.setitem(validate.ARCHIVE_TRUTH, f"other/{name}",
                            {"kind": "onset_within", "horizon_s": horizon})
    monkeypatch.setattr(validate, "_pooled_onset_rows", lambda *a, **kw: {
        "labels": rows, "shots_used": [1], "shots_with_onset": [1],
        "shots_in_training": [], "skipped": {}})
    report = validate.alarm_quality("other", [1], Paths.from_env(), thresholds=(.5,))
    if len(published) == 1:
        assert "horizon_integrated" not in report
        return
    # fnr 0 at both horizons: the integral of a zero rate is zero, and the
    # point is that it exists at all for a slug that is not the base one.
    assert set(report["horizon_integrated"]) == {"all", "held_out", "in_training"}
    assert report["horizon_integrated"]["held_out"]["0.5"]["fnr"] == 0.0


def test_pool_isolates_rejected_and_failed_matches(wired, monkeypatch):
    paths = Paths(root=wired['root'], corpus=wired['corpus'])
    paths.labels.mkdir(parents=True)
    for shot in [1, 2]:
        paths.labels_file(shot).touch()

    def match(shot, spec, paths, archive):
        if shot == 1:
            return validate._ShotMatch(skip_reason='match rejected: ambiguous')
        raise TimeoutError('shot budget expired')

    monkeypatch.setattr(validate, '_matched_shot', match)
    got = validate._pooled_onset_rows('d3d_tearing_time_to_event_dsm', [1, 2], paths,
                                     archive=wired['archive'], timeout_s=None)
    assert got['shots_used'] == []
    assert 'ambiguous' in got['skipped']['1']
    assert 'budget expired' in got['skipped']['2']
    assert got['labels']['tm_risk_1s']['y'].size == 0
    report = validate.alarm_quality('d3d_tearing_time_to_event_dsm', [], paths,
                                    archive=wired['archive'])
    for subset in ('all', 'held_out', 'in_training'):
        empty = report['labels']['tm_risk_1s']['subsets'][subset]
        assert empty['n_quiet'] == empty['n_tearing'] == empty['n_rows'] == 0
        rates = empty['thresholds']['0.5']
        assert rates['final_label']['fpr'] is None
        assert rates['warning_time_s']['median'] is None


def test_column_pool_keeps_archived_positives_and_whole_trace_alarm(wired, monkeypatch):
    from labelmaker.labels.schema import LabelSpec
    from labelmaker.labels.store import write_labels
    from labelmaker.models.base import Decoded

    slug = 'd3d_tearing_onset_cnn1d'
    paths = Paths(root=wired['root'], corpus=wired['corpus'])
    _fake_match(monkeypatch, tm_label=[0, 0, 0, 1, 1, 0])
    y = np.array([.1, .1, .1, .1, .9, .9, .1, .1])
    spec = LabelSpec(name='tm_prob', task='binary', activation='none', units='',
                     classes=(), slug=slug, card_id='test/model', time_step_ms=25,
                     ensemble_n=1, artifact_sha256='abc')
    write_labels(paths.labels_file(190000), 190000, _FakeBuilt().t,
                 {'tm_prob': Decoded(mean=y, lo=y, hi=y)}, (spec,), np.ones(8, bool),
                 run_id='test', features_sha256='abc')
    pooled = validate._pooled_onset_rows(slug, [190000], paths,
                                        archive=wired['archive'], timeout_s=None)
    rows = pooled['labels']['tm_prob']
    np.testing.assert_array_equal(rows['truth'], [0, 0, 0, 1, 1, 0])
    assert rows['t'].size == 6
    report = validate.alarm_quality(slug, [190000], paths,
                                    archive=wired['archive'], thresholds=(.5,))
    label = report['labels']['tm_prob']
    assert label['row_set'] == 'all valid rows of every aligned shot'
    assert label['ipcw_auc_reason'] == (
        'column label: the archived tm_label has no horizon, IPCW AUC is undefined')
    # The fake adapter names no training shots, so every row is held out.
    assert label['subsets']['in_training']['n_rows'] == 0
    held = label['subsets']['held_out']
    assert held['n_positive'] == 2 and held['auroc'] == 1
    assert held['ipcw_auc'] is None
    rates = label['subsets']['all']['thresholds']['0.5']
    assert rates['final_label']['counts']['FN'] == 1
    assert rates['any_row']['counts']['TP'] == 1


def test_empty_valid_population_has_no_alarm_verdict(monkeypatch):
    _fake_match(monkeypatch, tm_label=[0, 0, 0, 1, 1, 0])
    got = validate.score_against_truth(
        'd3d_tearing_time_to_event_dsm/tm_risk_1s', np.ones(8), np.zeros(8, bool),
        validate.archived_truth(190000, Paths.from_env()), threshold=.5)
    assert got['n'] == 0
    assert got['alarm'] is None and got['verdict'] is None


def test_grid_boundary_truth_is_shared_by_plain_and_ipcw_metrics(monkeypatch):
    from labelmaker import alarm

    t = .025 * np.array([1, 2, 3])
    onset = .025 * 12
    duration = onset - t
    assert duration[1] > .25  # 0.30000000000000004 - 0.05
    target = validate._onset_within_truth(t, onset, .25)
    np.testing.assert_array_equal(target, [False, True, True])
    truth = {'available': True, 'index': np.arange(3), 't': t, 'onset_s': onset}
    score = np.array([.5, .9, .1])
    got = validate.score_against_truth('d3d_tearing_time_to_event_dsm/tm_risk_250ms',
                                       score, np.ones(3, bool), truth, threshold=.5)
    assert got['n_positive'] == 2
    assert alarm.ipcw_auc(duration, np.ones(3, bool), score, .25,
                          cases=target) == got['auroc'] == .5
    # Exercise the report call site as well: dropping cases=target would
    # turn the high-scoring boundary case into a control and yield AUC 0.
    rows = {'shot': np.ones(3, int), 't': t, 'y': score, 'truth': target,
            'onset_s': np.full(3, onset), 't_end': np.full(3, onset),
            'in_training': np.zeros(3, bool), 'shots_used': [1], 'shots_with_onset': [1],
            'shots_in_training': [], 'skipped': {},
            'row_set': 'pre-onset valid rows of every aligned shot'}
    monkeypatch.setitem(validate.ARCHIVE_TRUTH, 'boundary/risk',
                        {'kind': 'onset_within', 'horizon_s': .25})
    monkeypatch.setattr(validate, '_pooled_onset_rows', lambda *a, **kw: {
        'labels': {'risk': rows}, 'shots_used': [1], 'shots_with_onset': [1],
        'shots_in_training': [], 'skipped': {}})
    report = validate.alarm_quality('boundary', [1], Paths.from_env(), thresholds=(.5,))
    everything = report['labels']['risk']['subsets']['all']
    assert everything['n_positive'] == 2
    assert everything['auroc'] == everything['ipcw_auc'] == .5


@pytest.mark.parametrize("onset", [0.1, None])
@pytest.mark.parametrize("ratios", [(2.0, 0.5), (2.0, 4.0)])
def test_time_to_onset_scores_only_positive_valid_pre_onset_predictions(monkeypatch, onset, ratios):
    _fake_match(monkeypatch, tm_label=[0, 0, 0, 1, 1, 0] if onset else [0] * N)
    truth = validate.archived_truth(190000, Paths.from_env())
    y = np.array([100.0 * ratios[0], 75.0 * ratios[1], 50000.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    valid = np.ones(8, bool)
    valid[2] = False
    got = validate.score_against_truth("d3d_tearing_time_to_event_dsm/tm_time_p50",
                                       y, valid, truth, threshold=0.5)
    if onset is None:
        assert got["scored"] is False and "onset" in got["reason"]
        return
    assert got["scored"] is True and got["n"] == 2
    assert got["kind"] == "time to archived onset (ms), tearing shots only"
    errors = np.log(ratios)
    assert got["median_abs_log_ratio"] == pytest.approx(np.median(np.abs(errors)))
    assert got["bias_log"] == pytest.approx(errors.mean(), abs=1e-12)
    assert got["rmse_log"] == pytest.approx(np.sqrt(np.mean(errors ** 2)))
    assert "alarm" not in got and "auroc" not in got


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_time_to_onset_drops_nonpositive_nonfinite_predictions(monkeypatch, bad):
    _fake_match(monkeypatch, tm_label=[0, 0, 0, 1, 1, 0])
    got = validate.score_against_truth("d3d_tearing_time_to_event_dsm/tm_time_p50",
                                       np.full(8, bad), np.ones(8, bool),
                                       validate.archived_truth(190000, Paths.from_env()),
                                       threshold=None)
    assert got["n"] == 0
    assert got["median_abs_log_ratio"] is got["bias_log"] is got["rmse_log"] is None


def test_calibration_study_splits_shots_and_publishes_only_all_rows(wired, monkeypatch):
    import json

    from labelmaker.calibrate import IsotonicMap, prior_shift

    slug = "d3d_tearing_time_to_event_dsm"
    paths = Paths(root=wired["root"], corpus=wired["corpus"])
    shots = np.repeat(np.arange(8), 4)
    onset = np.where(shots % 2 == 0, 2., np.nan)
    labels = {}
    for i, name in enumerate(("tm_risk_250ms", "tm_risk_500ms", "tm_risk_1s")):
        truth = (np.tile(np.arange(4), 8) <= i) & np.isfinite(onset)
        labels[name] = {"shot": shots, "t": np.tile(np.arange(4) * .1, 8),
                        "y": np.tile([.05, .1, .2, .3], 8), "truth": truth,
                        "onset_s": onset, "t_end": np.full(32, 3.),
                        "in_training": np.zeros(32, bool),
                        "shots_used": list(range(8)), "shots_with_onset": [0, 2, 4, 6],
                        "shots_in_training": [], "skipped": {}}
    pooled = {"labels": labels, "shots_used": list(reversed(range(8))),
              "shots_with_onset": [0, 2, 4, 6], "shots_in_training": [],
              "skipped": {"99": "missing"}}
    calls = []

    def pool(*args, **kwargs):
        calls.append(1)
        return pooled

    monkeypatch.setattr(validate, "_pooled_onset_rows", pool)
    source = {"t": {"path": "synthetic_t", "sha256": "abc"},
              "e": {"path": "synthetic_e", "sha256": "def"}}

    def reader():
        return np.array([250, 500, 1000, 1]), np.array([1, 1, 1, 0]), source

    report = validate.calibration_study(slug, list(range(8)) + [99], paths,
                                         training_reader=reader)
    assert len(calls) == 1
    fit = np.random.default_rng(0).permutation(np.arange(8))[:4].tolist()
    assert report["split"]["fit"] == fit
    assert set(report["split"]["report"]).isdisjoint(fit)
    assert report["training"]["files"] == source
    assert report["training"]["n_rows"] == 4
    assert report["prevalence_source"] == ("held-out FIT half of each row set; "
                                          "applied to every reported subset")
    saved = json.loads((paths.models / slug / "calibration.json").read_text())
    for i, (name, rows) in enumerate(labels.items()):
        item = report["labels"][name]
        assert item["q1"] == (i + 1) / 4
        for row_set in ("all_pre_onset", "onset_shots_only"):
            keep = np.ones(32, bool) if row_set == "all_pre_onset" else np.isfinite(onset)
            fitting = keep & np.isin(shots, fit)
            testing = keep & ~np.isin(shots, fit)
            result = item["row_sets"][row_set]
            assert result["p1"] == rows["truth"][fitting].mean()
            # No shot is a training shot here, so `all` and `held_out` are the
            # same population and `in_training` is empty.
            assert result["subsets"]["in_training"]["n_rows"] == 0
            iso = IsotonicMap.fit(rows["y"][fitting], rows["truth"][fitting])
            values = {"raw": rows["y"][testing],
                      "prior_shift": prior_shift(rows["y"][testing], from_prevalence=item["q1"],
                                                 to_prevalence=result["p1"]),
                      "isotonic": iso.apply(rows["y"][testing])}
            for method, prob in values.items():
                want = validate.binary_metrics(prob, rows["truth"][testing])
                for subset in ("all", "held_out"):
                    got = result["subsets"][subset]["methods"][method]
                    for key in ("n", "n_positive", "ece", "brier", "auroc", "calibration"):
                        assert got[key] == want[key]
                    assert len(got["calibration"]) == 10
        published = saved["labels"][name]
        fit_on = published["fit_on"]
        assert fit_on["row_set"] == "all_pre_onset"
        assert fit_on["subset"] == "held_out"
        assert fit_on["shots"] == sorted(fit)
        assert fit_on["n_rows"] == 16
        assert fit_on["prevalence"] == item["row_sets"]["all_pre_onset"]["p1"]
        assert fit_on["date"] and fit_on["git_sha"]
        assert published["map"]["n_fit"] == 16
    assert json.loads((paths.validation / slug / "calibration_study.json").read_text()) == report


def test_calibration_fits_on_held_out_shots_and_reports_three_ways(wired, monkeypatch):
    """The published map may not be fitted on shots the model was trained on.

    An isotonic map fitted on memorised rows is calibrated to memorisation,
    so the fit half is drawn from the held-out shots alone; the in-training
    shots are reported beside them, never mixed into the fit.
    """
    import json

    from labelmaker.calibrate import IsotonicMap

    slug, name = "d3d_tearing_time_to_event_dsm", "tm_risk_1s"
    paths = Paths(root=wired["root"], corpus=wired["corpus"])
    shots = np.repeat(np.arange(8), 4)
    onset = np.where(shots % 2 == 0, 2., np.nan)
    in_training = shots < 4
    truth = (np.tile(np.arange(4), 8) <= 1) & np.isfinite(onset)
    rows = {"shot": shots, "t": np.tile(np.arange(4) * .1, 8),
            "y": np.tile([.05, .1, .2, .3], 8), "truth": truth,
            "onset_s": onset, "t_end": np.full(32, 3.), "in_training": in_training,
            "shots_used": list(range(8)), "shots_with_onset": [0, 2, 4, 6],
            "shots_in_training": [0, 1, 2, 3], "skipped": {}}
    monkeypatch.setattr(validate, "_pooled_onset_rows", lambda *a, **k: {
        "labels": {name: rows}, "shots_used": list(range(8)),
        "shots_with_onset": [0, 2, 4, 6], "shots_in_training": [0, 1, 2, 3], "skipped": {}})
    report = validate.calibration_study(
        slug, list(range(8)), paths,
        training_reader=lambda: (np.array([1000., 1.]), np.array([1, 0]), {}))

    held_out = np.random.default_rng(0).permutation([4, 5, 6, 7])
    fit, reported = held_out[:2].tolist(), held_out[2:].tolist()
    assert report["split"] == {"fit": fit, "report": reported, "in_training": [0, 1, 2, 3],
                               "fit_subset": "held_out"}
    everything = report["labels"][name]["row_sets"]["all_pre_onset"]
    assert everything["fit_subset"] == "held_out"
    fitting = np.isin(shots, fit)
    iso = IsotonicMap.fit(rows["y"][fitting], truth[fitting])
    assert everything["p1"] == iso.prevalence_fit and everything["n_fit"] == iso.n_fit == 8
    counts = {key: (s["n_shots"], s["n_rows"]) for key, s in everything["subsets"].items()}
    # `all` is every shot the fit did not use: the report half plus every
    # in-training shot.
    assert counts == {"all": (6, 24), "held_out": (2, 8), "in_training": (4, 16)}
    for key, subset in everything["subsets"].items():
        keep = np.isin(shots, {"all": reported + [0, 1, 2, 3], "held_out": reported,
                               "in_training": [0, 1, 2, 3]}[key])
        want = validate._calibration_metrics(iso.apply(rows["y"][keep]), truth[keep])
        assert subset["methods"]["isotonic"] == want
        assert subset["n_tearing"] + subset["n_quiet"] == subset["n_shots"]
    onset_only = report["labels"][name]["row_sets"]["onset_shots_only"]
    assert onset_only["subsets"]["held_out"]["n_quiet"] == 0
    fit_on = json.loads((paths.models / slug / "calibration.json").read_text())
    assert fit_on["labels"][name]["fit_on"]["subset"] == "held_out"
    assert fit_on["labels"][name]["fit_on"]["shots"] == sorted(fit)


@pytest.mark.parametrize("truth, brier, ece", [([0, 0], .34, .5), ([1, 1], .34, .5)])
def test_calibration_metrics_keep_single_class_scores(truth, brier, ece):
    result = validate._calibration_metrics(np.array([.2, .8]), np.array(truth))
    assert result["auroc"] is None
    assert result["n"] == 2
    assert result["brier"] == pytest.approx(brier)
    assert result["ece"] == pytest.approx(ece)
    assert len(result["calibration"]) == 10


def test_calibration_empty_fit_preserves_existing_map(wired, monkeypatch):
    slug = "d3d_tearing_time_to_event_dsm"
    paths = Paths(root=wired["root"])
    saved = paths.models / slug / "calibration.json"
    saved.parent.mkdir(parents=True)
    saved.write_text("existing map")
    monkeypatch.setattr(validate, "_pooled_onset_rows", lambda *a, **k: {
        "labels": {}, "shots_used": [], "shots_with_onset": [],
        "shots_in_training": [], "skipped": {}})
    with pytest.raises(ValueError, match="at least two held-out aligned shots"):
        validate.calibration_study(slug, [], paths,
                                   training_reader=lambda: pytest.fail("must not read training"))
    assert saved.read_text() == "existing map"


@pytest.mark.parametrize("row_set", ["all_pre_onset", "onset_shots_only"])
def test_calibration_empty_population_names_label_and_row_set(wired, monkeypatch, row_set):
    slug, name = "d3d_tearing_time_to_event_dsm", "tm_risk_250ms"
    paths = Paths(root=wired["root"])
    saved = paths.models / slug / "calibration.json"
    saved.parent.mkdir(parents=True)
    saved.write_text("existing map")
    fit, report = np.random.default_rng(0).permutation([1, 2])
    shots = np.array([report]) if row_set == "all_pre_onset" else np.array([fit, report])
    rows = {"shot": shots, "y": np.full(shots.size, .2),
            "truth": np.zeros(shots.size), "onset_s": np.full(shots.size, np.nan),
            "in_training": np.zeros(shots.size, bool)}
    monkeypatch.setattr(validate, "_pooled_onset_rows", lambda *a, **k: {
        "labels": {name: rows}, "shots_used": [1, 2], "shots_in_training": [],
        "skipped": {}})
    with pytest.raises(ValueError, match=f"{name}: empty fitting population for {row_set}"):
        validate.calibration_study(slug, [1, 2], paths,
                                   training_reader=lambda: (np.array([1]), np.array([1]), {}))
    assert saved.read_text() == "existing map"
