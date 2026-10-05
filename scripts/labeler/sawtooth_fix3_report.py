"""Render one current-state, source-backed sawtooth report and repository docs."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from sawtooth_physics import OUTPUT, REPO, WORK

from labeler.sawtooth.preprocessing import STATES

NAMES = {
    "saw-derivative": "Single-channel derivative / ±125 ms presence",
    "saw-always-present": "Always present",
    "saw-hl3": "Adapted HL-3 / derivative picker gated by HL-3",
    "saw-ours": "PhaseNet-style picker",
}


def number(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def interval(value, ci):
    return number(value) + (f" [{number(ci[0])}, {number(ci[1])}]" if ci else "")


def distribution(summary, unit=""):
    """Render a quantile summary as text, not as a raw dictionary."""
    levels = "/".join(f"{100 * q:g}" for q in summary["quantile_levels"])
    values = ", ".join(number(v) for v in summary["quantiles"])
    text = (
        f"n={summary['n']}; mean {number(summary['mean'])}{unit}; "
        f"quantiles at {levels}%: {values}{unit}"
    )
    if summary.get("within_0p15_m_fraction") is not None:
        text += f"; {100 * summary['within_0p15_m_fraction']:.1f}% within 0.15 m"
    return text


def seconds_text(seconds):
    return ", ".join(
        f"{state} {number(value, 2)} s" for state, value in seconds.items()
    )


def limitation_lines(work, data, check, bench, composition, gaps, radial, side):
    """Known limitations stated from the records, not from memory."""
    population = data["population"]
    candidates = population["candidate_state_counts"]
    seconds = population["state_seconds"]
    observable = population["observable_seconds"]
    example = json.loads((work / "shots" / "203563.json").read_text())
    states = Counter(p["attrs"]["state"] for p in example["crashes"])
    first = example["crashes"][0]["attrs"]
    cohort_tested = sum(
        composition["after"][k]["tested_absent_s"] for k in ("train", "val", "test")
    )
    cohort_observable = sum(
        data["splits"][k]["observable_seconds"] for k in ("train", "val", "test")
    )
    rule_observable = {
        r["shot"]: r["frozen_rule"]["states"]["observable_s"] for r in check["by_shot"]
    }
    return [
        (
            "Known limitations, all left as they are because a rule change would "
            "force a full population rerun:\n"
        ),
        (
            f"- Uncertain dominates: {100 * seconds['uncertain'] / observable:.1f}% of "
            f"observable population time and {candidates['uncertain']} of "
            f"{candidates['present'] + candidates['uncertain']} candidates are "
            f"uncertain. Present plus tested absent is "
            f"{100 * population['assessed_fraction_of_observable']:.1f}% of "
            "observable time, so all conditional scores describe that fraction."
        ),
        (
            f"- The negative class is thin: tested absence is "
            f"{100 * cohort_tested / cohort_observable:.1f}% of observable cohort "
            "time, so the assessed set is positive-heavy and presence AUROC and "
            "precision are less informative than before; the old-negatives "
            "sensitivity table is the comparison to the earlier scoring."
        ),
        (
            "- The ECE-validity step test reads a static calibration step as an "
            "invalid profile. On the Muscatello reference shots it removes "
            f"nearly all time (observable {number(rule_observable[141182], 2)} s "
            f"and {number(rule_observable[141195], 2)} s of 3.1 s), so the frozen "
            "rule labels almost nothing on them. It is the review's test and it "
            "is not tuned on those shots."
        ),
        (
            "- The EFIT01 q=1 radius conflict is conservative. Shot 203563 "
            f"(EFIT01 q_min {first['qmin']:.2f}, inversion minus q=1 radius "
            f"{first['q1_radius_difference_m']:.3f} m) has "
            f"{states['uncertain']} of {len(example['crashes'])} candidates "
            "uncertain although it shows regular trains. The paired comparison "
            f"is {100 * side['low_field_side_fraction']:.1f}% low-field side."
        ),
        (
            f"- Uncertain gaps inside present trains cover "
            f"{100 * gaps['gap_share_of_block_time']:.1f}% of block time on the "
            "cohort; most are missed crashes that break the period-ratio test."
        ),
        (
            "- Population shots without archived field or axis metadata, and "
            "shots without a usable ECE waveform, receive no definite label "
            "(see the exclusion counts above)."
        ),
        (
            f"- The reviewed 186636 span is {100 * (1 - radial['shots']['186636']['support']['observable_fraction']):.0f}% "
            "cutoff or invalid ECE and is untestable here; the 190637 span is "
            "not a central sawtooth by ECE. Neither is a calibrated "
            "physical truth."
        ),
        (
            "- The legacy-offset diagnosis traces the old detector's offsets to "
            "its inversion-profile gate and holdoff, not to the reader. The "
            "current catalog detector could be compared only on shot 192090."
        ),
        (
            "- Each gallery crash panel shows the accepted crash with the "
            "largest A_norm (present crashes first; an uncertain crash only when "
            "the shot has no present one), a fixed rule, so panels are not "
            "chosen to be easy or hard."
        ),
        "",
    ]


def source(name, key=""):
    suffix = f" → `{key}`" if key else ""
    return f"Source: `outputs/labeler/sawtooth/fix5/{name}`{suffix}.\n"


def timing_table(legacy):
    lines = [
        (
            "| Shot / split | Prior offset ms | Fresh offset ms (pairs) | "
            "Fresh present picks | Observable s | Derivative matches: "
            "legacy / no holdoff |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in legacy["timing_table"]:
        pairs = row["fresh_offset_pairs_within_15ms"]
        fresh = number(row["fresh_old_minus_physics_ms_within_15ms"], 2)
        lines.append(
            f"| {row['shot']} / {row['split']} | "
            f"{number(row['legacy_minus_prior_ms'], 2)} | {fresh} ({pairs}) | "
            f"{row['fresh_present_points']} | "
            f"{number(row['fresh_observable_seconds'])} | "
            f"{number(row['legacy_matches_derivative_2ms'], 0)} / "
            f"{number(row['no_holdoff_matches_derivative_2ms'], 0)} |"
        )
    return "\n".join(lines) + "\n"


def reference_table(reference):
    lines = [
        (
            "| Shot | Window s | Period ms | Published period ms | "
            "Relative amplitude | Published amplitude | Status |"
        ),
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in reference["by_shot"]:
        period, amplitude = row["published_period_ms"], row["published_amplitude"]
        published = (
            f"{period[0]} ± {period[1]}",
            f"{amplitude[0]} ± {amplitude[1]}",
        )
        if not row.get("windows"):
            lines.append(
                f"| {row['shot']} | — | — | {published[0]} | — | "
                f"{published[1]} | {row['status']} |"
            )
        for window in row.get("windows", []):
            left, right = window["window_s"]
            lines.append(
                f"| {row['shot']} | {left}–{right} | "
                f"{number(window['median_period_ms'], 2)} | {published[0]} | "
                f"{number(window['median_amplitude'])} | {published[1]} | "
                f"{row['status']} |"
            )
    return "\n".join(lines) + "\n"


def model_table(models, *, fixed=False):
    """Presence leads with the threshold-free AUPRC; F1 is shown at two thresholds."""
    lines = [
        (
            "| Method | Presence AUPRC [95% CI] | Presence AUROC [95% CI] | "
            "Presence F1, fixed 0.5 [95% CI] | Presence F1, inner-selected "
            "threshold [95% CI] | Crash F1 ±2 ms [95% CI] | "
            "Assessed / excluded / observable picks |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, label in NAMES.items():
        result = models[name]["fixed_validation"] if fixed else models[name]
        stats = result["crash_tolerance_2ms"]
        crash, presence, ci = stats["crash"], stats["presence"], stats["ci95"]
        counts = result["assessment_totals"]
        crash_f1 = interval(crash["f1"], ci.get("crash_f1")) if crash else "—"
        pooled = result.get("presence_fixed_threshold")
        at_fixed = (
            interval(pooled["pooled"]["f1"], pooled["ci95"].get("presence_f1"))
            if pooled
            else "—"
        )
        lines.append(
            f"| {label} | "
            f"{interval(presence['auprc'], ci.get('presence_auprc'))} | "
            f"{interval(presence['auroc'], ci.get('presence_auroc'))} | "
            f"{at_fixed} | "
            f"{interval(presence['f1'], ci.get('presence_f1'))} | {crash_f1} | "
            f"{counts['assessed_picks']} / {counts['excluded_picks']} / "
            f"{counts['observable_picks']} |"
        )
    return "\n".join(lines) + "\n"


def benchmark_framing(models):
    """What the conditional benchmark can and cannot show, from its own numbers."""
    derivative = models["saw-derivative"]
    ours = models["saw-ours"]
    d_crash = derivative["crash_tolerance_2ms"]["crash"]["f1"]
    o_crash = ours["crash_tolerance_2ms"]["crash"]["f1"]
    totals = ours["assessment_totals"]
    comparison = (
        f"the derivative baseline's crash F1 ({number(d_crash)}) is higher than "
        f"saw-ours ({number(o_crash)})"
        if d_crash > o_crash
        else f"the derivative baseline's crash F1 ({number(d_crash)}) is not "
        f"higher than saw-ours ({number(o_crash)}), although it starts with an "
        "advantage"
    )
    return [
        (
            "**What this benchmark can and cannot show.** Presence is a sanity "
            "check, close to trivial: the single-channel derivative baseline "
            "reaches AUROC "
            f"{number(derivative['crash_tolerance_2ms']['presence']['auroc'])} "
            "out of fold and "
            f"{number(derivative['fixed_validation']['crash_tolerance_2ms']['presence']['auroc'])} "
            "on the fixed validation shots, so a model has little room to "
            "separate itself. The labels are built from the same ECE edges the "
            f"derivative baseline reads, so {comparison}: crash F1 here is "
            "agreement with the rule, not physical accuracy. Only "
            f"{totals['both_class_shots']} of the {totals['assessed_shots']} "
            "assessed out-of-fold shots have both present and tested-absent "
            "bins, so most per-shot presence scores rest on one class. The blind "
            "expert queue is the real test of the labels and the models; nothing "
            "here replaces it. Presence is led by the threshold-free AUPRC, with "
            "F1 at one fixed threshold (0.5) beside the F1 at the inner-selected "
            "one.\n"
        ),
    ]


def fixed_threshold_table(models):
    """Per-fold presence scores at one fixed threshold and at the inner-selected one."""
    lines = [
        (
            "| Method | Fold | Held-out shots | Inner-selected threshold | "
            "Presence F1 at the selected threshold | Presence F1 at 0.5 | "
            "AUPRC | Positive / negative bins |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in ("saw-hl3", "saw-ours"):
        for row in models[name]["presence_fixed_threshold"]["by_fold"]:
            selected, fixed = row["at_selected_threshold"], row["at_fixed_threshold"]
            lines.append(
                f"| {NAMES[name]} | {row['fold']} | {row['shots']} | "
                f"{number(row['selected_threshold'])} | {number(selected['f1'])} | "
                f"{number(fixed['f1'])} | {number(fixed['auprc'])} | "
                f"{row['positive_bins']:,} / {row['negative_bins']:,} |"
            )
    return "\n".join(lines) + "\n"


def hours(seconds):
    return f"{seconds / 3600:.2f} h"


def method_lines(rule, derivation):
    """The detector and state definitions, from the frozen rule values."""
    return [
        (
            "The detector reads ECE channels 1–40 at 10 kHz (native-rate "
            "antialiasing, then decimation), the EFIT01 axis, boundary and q "
            "profile, and optional D-alpha, Mirnov, neutron, NBI and Ip traces. "
            "Every number below is in `freeze.json` (rule values, the 16 "
            "development shots, the excluded reviewed shots), which is hashed "
            "beside the labels.\n"
        ),
        (
            f"1. **Observability.** A sample is observable when at least "
            f"{rule['minimum_channels']} core channels (nearest the EFIT axis) "
            f"read at least {rule['te_floor_kev']} keV, and unobservable where "
            "R<(2/3)R_LCFS,out (third-harmonic overlap), where the Thomson "
            "density exceeds 0.9 of the X2 cutoff 2(f_ce/8.98 GHz)² or where the "
            f"ECE-validity test holds for at least "
            f"{rule['ece_validity_sustain_ms']:g} ms on a "
            f"{rule['ece_validity_smooth_ms']:g} ms moving mean: an "
            f"adjacent-channel step ratio above {rule['ece_step_ratio']:g} "
            f"inside nominal ρ<{rule['ece_step_max_rho']:g}, or the channel "
            f"within {rule['ece_axis_max_distance_m']:g} m of the axis below "
            f"{rule['ece_axis_to_max']:g} of the profile maximum. f_ce in the "
            "cutoff comes from the local field at the EFIT axis resonance "
            "(F/R_axis) wherever that axis field is mapped, with or without a "
            "Bt trace; the field Bt(R0) is used only where the axis field is "
            "unmapped but Bt exists, and a fixed 8×10¹⁹ m⁻³ guard only where "
            "neither exists (the record's `cutoff_field` names the branch). A "
            "channel whose record median is below the "
            f"{rule['te_floor_kev']} keV floor, or that shows no fluctuation, is "
            "a dead channel and is left out of both validity tests: a cutoff "
            "step is one-sided and a dead channel is low on both sides.\n"
        ),
        (
            f"2. **Edge filter and POSR.** Each channel is filtered with a "
            f"Gaussian first derivative (σ={rule['sigma_ms']} ms), and a local "
            "maximum of its absolute value is a candidate edge when the step "
            "is at least 0.5% of the local Te and its POSR reaches "
            f"{rule['posr_threshold']:g}. POSR is Gude's: the peak's distance "
            f"from the mean of a {rule['frame_ms']} ms frame, in standard "
            "deviations of that frame after dropping its ⌈7σ⌉ largest "
            "absolute values (the kernel length). The simulated "
            "noise-frame rate is in `noise_calibration.json`.\n"
        ),
        (
            f"3. **Multichannel coincidence.** Candidates within "
            f"{rule['coincidence_ms']} ms form a cluster. A cluster needs at "
            f"least {rule['minimum_channels']} channels and an observable core, "
            "and the next cluster is dropped inside the same holdoff.\n"
        ),
        (
            "4. **Inversion profile (Gude's A_norm and A_net).** From the "
            "per-channel step across the crash, valid channels hotter than "
            f"{rule['maximum_channel_to_core']}× the local core level are "
            "masked. A_norm=Σ|step|/ΣTe must reach "
            f"{rule['significance']}; A_net=|Σstep|/Σ|step| must stay below "
            f"{rule['maximum_net']}, which rejects a profile that falls "
            f"everywhere. A contiguous loss block of at least "
            f"{rule['minimum_block']} channels (steps below −0.5% of local Te) "
            f"needs a contiguous gain block of at least {rule['minimum_block']} "
            f"channels within {rule['pulse_reach']} channels. The block "
            "boundary is the inversion channel.\n"
        ),
        (
            f"5. **Central drop and edge rejection.** At the channel nearest "
            "the nominal EFIT axis the relative drop (mean over 0.3–1.5 ms "
            "after against 0.3–1.5 ms before) must be at least "
            f"{rule['central_relative_drop']}. A crash whose loss is outside the "
            f"core and that coincides with a D-alpha burst (z≥"
            f"{rule['dalpha_burst_z']:g}) is rejected as an ELM or edge event; "
            f"Ip below {rule['minimum_ip_ma']} MA is rejected. Neutron and "
            "Mirnov bursts corroborate when present and never downgrade an ECE "
            "crash.\n"
        ),
        (
            f"6. **Trains.** At least {rule['minimum_train']} accepted crashes "
            f"with gaps of {rule['minimum_period_ms']:g}–"
            f"{rule['maximum_period_ms']:g} ms, successive-gap ratio at most "
            f"{rule['period_ratio']}, and an inversion channel spread within "
            f"{rule['inversion_spread_channels']:g} channels form a present "
            "train; trains split at every unobservable sample.\n"
        ),
        (
            f"7. **Uncertainty reasons.** A train crash is `uncertain` when "
            f"EFIT01 q_min exceeds {1 + rule['qmin_margin']:g}, when the nominal "
            f"inversion R is more than {rule['radius_tolerance']} m from the "
            "same-side EFIT01 q=1 R, or when the inversion has no nominal R. "
            "Geometry is the nominal second-harmonic vacuum resonance, not flux.\n"
        ),
        (
            "8. **States.** `present` is a train span. `absent` is **tested "
            "absence**: no POSR-periodic edge (3 or more, 20–250 ms) on any "
            f"valid channel at nominal ρ<{rule['quiet_core_max_rho']:g} over a "
            "complete ±375 ms observable context, noise-resolved on at least two "
            "channels, with no profile candidate or slow relaxation phase "
            f"nearby and no isolated edge within "
            f"{rule['isolated_edge_context_ms']:g} ms. An edge counts against "
            "absence only when its relative change reaches "
            f"`significance` = {100 * rule['significance']:g}% of the local Te, so "
            f"a core relaxation train below {100 * rule['significance']:g}% is "
            "called quiet. The isolated-edge length was derived on TRAIN shots "
            f"only ({derivation['derivation_shot_count']} shots, "
            "`edge_context_derivation.json`): it is the "
            f"{derivation['criterion']}. Time supported only by sustained "
            f"EFIT01 q_min≥{rule['qmin_absence']} ({rule['qmin_sustain_ms']:g} "
            "ms) is never absent. It is `q_prior_ece_contradicted` where it lies "
            "within the absence-test context of a periodic edge or a profile "
            "candidate, which is where the ECE shows relaxation evidence, and "
            "`q_prior_untested` elsewhere, where the ECE test did not run or "
            "was inconclusive. Both are exported as `uncertain` with the state "
            "name as the reason and are never a benchmark negative. "
            "`uncertain` is observable time without definite evidence and "
            "`unassessed` is unobservable time. `assessed` means present or "
            "absent. High q is neither necessary nor sufficient for absence: a "
            "q_min≥1.5 shot can still show a sawtooth-like core relaxation.\n"
        ),
    ]


def composition_lines(composition):
    """The absent class before and after the tested-absence policy."""
    lines = [
        (
            "| Set | Absent before s | High-q share before | Tested absent s | "
            "Q-prior, ECE-contradicted s | Q-prior, untested s | "
            "ECE-contradicted share of q-prior | "
            "Tested share of former absent class | Shots with tested absence |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("train", "val", "test", "population"):
        before, after = composition["before"][name], composition["after"][name]
        share = after["q_prior_ece_contradicted_fraction"]
        lines.append(
            f"| {name} | {number(before['absent_s'], 1)} | "
            f"{100 * before['high_q_fraction_of_absent']:.1f}% | "
            f"{number(after['tested_absent_s'], 1)} | "
            f"{number(after['q_prior_ece_contradicted_s'], 1)} | "
            f"{number(after['q_prior_untested_s'], 1)} | "
            f"{'—' if share is None else f'{100 * share:.1f}%'} | "
            f"{100 * after['tested_fraction_of_former_absent_class']:.1f}% | "
            f"{after['shots_with_tested_absence']} of {after['shots']} |"
        )
    return "\n".join(lines) + "\n"


def context_lines(composition, models):
    """Tested absence and benchmark negatives under three isolated-edge vetoes."""
    sensitivity = composition["edge_context_sensitivity"]
    own = sensitivity["rule_value_ms"]
    contexts = [k for k in sensitivity["train"]]
    lines = [
        (
            "| Isolated-edge veto ms | Train tested absent s | Val tested absent s | "
            "Test tested absent s | Tested share of observable (train) | "
            "OOF benchmark negative bins | OOF positive bins |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ours = models["saw-ours"]["edge_context_sensitivity"]
    for key in contexts:
        row = ours[f"{key}_ms"]
        mark = " (rule value)" if abs(float(key) - own) < 1e-9 else ""
        lines.append(
            f"| {key}{mark} | "
            + " | ".join(
                number(sensitivity[split][key]["tested_absent_s"], 1)
                for split in ("train", "val", "test")
            )
            + f" | {100 * sensitivity['train'][key]['fraction_of_observable']:.1f}% | "
            f"{row['negative_bins']:,} | {row['positive_bins']:,} |"
        )
    return "\n".join(lines) + "\n"


def guard_status_lines(composition):
    """Records by the density-guard branch they took, and dead channels found."""
    names = {
        "Thomson_90percentile_and_local_axis_field": "local axis field",
        "Thomson_90percentile_and_reference_bt_axis_unmapped": (
            "Bt(R0), axis field unmapped"
        ),
        "Thomson_90percentile_fixed_density_guard_axis_field_and_bt_unavailable": (
            "fixed 8e19 guard (no axis field, no Bt)"
        ),
    }
    lines = [
        (
            "| Set | Records | Local axis field | Bt(R0), axis field unmapped | "
            "Fixed 8e19 guard | No Thomson density | Records with dead channels | "
            "Dead channels |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("train", "val", "test", "population"):
        guards = composition["guards"][name]
        counts = guards["records_by_density_guard_status"]
        known = [counts.get(key, 0) for key in names]
        total = sum(counts.values())
        lines.append(
            f"| {name} | {total} | "
            + " | ".join(str(count) for count in known)
            + f" | {total - sum(known)} | {guards['records_with_dead_channels']} | "
            f"{guards['dead_channels_total']} |"
        )
    return "\n".join(lines) + "\n"


def guard_lines(composition):
    lines = [
        (
            "| Set | Core-observable s | Removed by the earlier guard (Bt at R0, "
            "fixed 8e19 without Bt) | "
            "Removed by the current guard | Removed by ECE validity | "
            "Shots with >20% removed by validity |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("train", "val", "test", "population"):
        guards = composition["guards"][name]
        seconds, fraction = guards["seconds"], guards["fraction_of_core_observable"]
        lines.append(
            f"| {name} | {number(seconds['core_observable_samples'], 1)} | "
            f"{number(seconds['removed_by_reference_density_guard'], 1)} "
            f"({100 * fraction['removed_by_reference_density_guard']:.1f}%) | "
            f"{number(seconds['removed_by_density_guard'], 1)} "
            f"({100 * fraction['removed_by_density_guard']:.1f}%) | "
            f"{number(seconds['removed_by_ece_validity'], 1)} "
            f"({100 * fraction['removed_by_ece_validity']:.1f}%) | "
            f"{guards['shots_with_ece_validity_above_20_percent']} of {guards['shots']} |"
        )
    return "\n".join(lines) + "\n"


def sensitivity_lines(models):
    """Scores with the previous absent class restored as scoring-only negatives."""
    lines = [
        (
            "| Held-out set | Method | Presence F1 now | Presence F1, previous "
            "negatives | AUROC now | AUROC, previous negatives | Crash F1 ±2 ms "
            "now | Crash F1, previous negatives |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for fixed, label in ((False, "OOF"), (True, "Fixed validation")):
        for name, text in NAMES.items():
            row = models[name]["fixed_validation"] if fixed else models[name]
            now = row["crash_tolerance_2ms"]
            old = row["old_negatives_sensitivity"]["crash_tolerance_2ms"]
            crash_now = now["crash"]["f1"] if now["crash"] else None
            crash_old = old["crash"]["f1"] if old["crash"] else None
            lines.append(
                f"| {label} | {text} | {number(now['presence']['f1'])} | "
                f"{number(old['presence']['f1'])} | {number(now['presence']['auroc'])} | "
                f"{number(old['presence']['auroc'])} | {number(crash_now)} | "
                f"{number(crash_old)} |"
            )
    return "\n".join(lines) + "\n"


def operating_table(models):
    """Per-fold operating points selected on inner shots, and their spread."""
    lines = [
        (
            "| Method | Fold | Presence threshold | Crash threshold | Crash z | "
            "Inner shots | Best / completed epochs |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("saw-hl3", "saw-ours"):
        points = models[name]["operating_points"]
        for row in points["by_fold"]:
            lines.append(
                f"| {NAMES[name]} | {row['fold']} | "
                f"{number(row['presence_threshold'])} | "
                f"{number(row['crash_threshold'])} | {number(row['crash_z'], 2)} | "
                f"{row['selection_shots']} | "
                f"{row['best_epoch']} / {row['epochs_completed']} |"
            )
        lines.append(
            f"| {NAMES[name]} | spread | "
            f"{number(points['presence_threshold_min'])}–"
            f"{number(points['presence_threshold_max'])} (range "
            f"{number(points['presence_threshold_range'])}) | | | | |"
        )
    return "\n".join(lines) + "\n"


def rule_reference_table(check):
    """Frozen rule and derivative picker against the published Muscatello bands."""
    lines = [
        (
            "| Shot | Window s | Expected crashes | Method | Found | Period ms "
            "(85 ± 5) | Amplitude (0.35 ± 0.02) | Unconfirmed / unpicked |"
        ),
        "|---|---|---:|---|---:|---|---|---|",
    ]
    mark = {True: "pass", False: "FAIL", None: "no crash"}
    for row in check["by_shot"]:
        for key, name in (
            ("derivative_picker", "derivative picker"),
            ("frozen_rule", "frozen rule"),
            ("rule_validity_test_off", "frozen rule, validity test off (diagnostic)"),
        ):
            for window in row[key]["windows"]:
                left, right = window["window_s"]
                low, high = window["crashes_expected_range"]
                agreement = window.get("agreement_with_picker")
                counts = (
                    f"{len(agreement['unconfirmed_picks_s'])} / "
                    f"{len(agreement['unpicked_reference_s'])}"
                    if agreement
                    else "—"
                )
                lines.append(
                    f"| {row['shot']} | {left}–{right} | {low}–{high} | {name} | "
                    f"{window['crashes_found']} | "
                    f"{number(window['median_period_ms'], 1)} "
                    f"({mark[window['period_pass']]}) | "
                    f"{number(window['median_amplitude'])} "
                    f"({mark[window['amplitude_pass']]}) | {counts} |"
                )
    return "\n".join(lines) + "\n"


def gap_lines(gaps):
    """Why uncertain gaps appear inside otherwise clean trains, and their share."""
    kinds, seconds = gaps["gap_kinds"], gaps["gap_seconds_by_kind"]
    reasons = ", ".join(
        f"{count} × {name}" for name, count in gaps["uncertain_crash_reasons"].items()
    )
    example = gaps["example"]
    durations = [g["end_s"] - g["start_s"] for g in example["gaps"]]
    listing = "; ".join(
        f"{g['start_s']:.2f}–{g['end_s']:.2f} s ({g['kind'].replace('_', ' ')})"
        for g in example["gaps"]
    )
    return [
        (
            "A present train is split wherever a crash is missing or uncertain, "
            "and the gap between present spans is exported `uncertain`. Joining "
            f"present spans across gaps of at most {gaps['max_gap_s']} s that are "
            "entirely uncertain gives "
            f"{gaps['blocks_with_uncertain_gaps']} blocks with "
            f"{gaps['gaps']} gaps on the cohort. The gaps are "
            f"{number(gaps['uncertain_gap_seconds'], 1)} s, "
            f"{100 * gaps['gap_share_of_block_time']:.1f}% of block time and "
            f"{100 * gaps['gap_share_of_observable_time']:.1f}% of observable time. "
            f"{kinds.get('no_accepted_crash', 0)} gaps "
            f"({number(seconds.get('no_accepted_crash', 0), 1)} s) contain no "
            "accepted crash: a crash was rejected by the profile, central-drop or "
            "coincidence tests, or fell inside the holdoff, and the following "
            "gap breaks the period-ratio test; the span is then neither present "
            "nor tested absent. "
            f"{kinds.get('uncertain_crash', 0)} gaps "
            f"({number(seconds.get('uncertain_crash', 0), 1)} s) contain an "
            f"accepted crash left uncertain ({reasons}).\n"
        ),
        (
            f"Shot {example['shot']} has {len(example['gaps'])} such gaps "
            f"totalling {number(example['uncertain_s'], 2)} s: {listing}. "
            f"Durations run {1000 * min(durations):.0f}–{1000 * max(durations):.0f} ms "
            "against the shot's median accepted-crash period of about 50 ms, so "
            "most are one or two missed crashes and the longest is about five.\n"
        ),
        source("uncertain_gaps.json"),
    ]


GATE_TEXT = {
    "found": "found by the rule",
    "train_test": "passed every gate but is not part of a stable train",
    "no_edge_cluster": "no two-channel POSR edge cluster at the crash",
    "coincidence": "only one channel reaches the POSR threshold",
    "unobservable_core": "core not observable at the crash",
    "holdoff": "inside the holdoff of the previous cluster",
    "coverage": "too few valid channels in the inversion profile",
    "significance": "profile A_norm below the significance threshold",
    "redistribution": "profile falls or rises everywhere (A_net)",
    "drop_block": "no contiguous loss block of two channels",
    "rise_block": "no adjacent gain block of two channels",
    "central_relative_drop": "central drop below the floor",
    "unobservable_central_channel": "central channel unobservable",
    "elm_edge_only": "edge loss with a D-alpha burst",
    "edge_only_proxy": "loss outside the core",
    "low_plasma_current": "plasma current below the floor",
    "calibrated_direction": "calibrated flux direction conflict",
}


def gate_text(counts):
    return "; ".join(
        f"{count} × {GATE_TEXT.get(gate, gate)}"
        for gate, count in sorted(counts.items(), key=lambda kv: -kv[1])
    )


def reference_gate_lines(check):
    """Which gate decides each published reference crash, for both rule rows."""
    rows = {row["shot"]: row for row in check["by_shot"]}
    lines = []
    for shot in (141182, 141195):
        row = rows[shot]
        frozen = row["frozen_rule"]["reference_crash_gates"]["counts"]
        off = row["rule_validity_test_off"]["reference_crash_gates"]["counts"]
        lines.append(
            f"- Shot {shot}: frozen rule, {gate_text(frozen)}. "
            f"With the validity test off: {gate_text(off)}."
        )
    return [
        (
            "Which gate decides the reference crashes. Each derivative-picker "
            "crash inside the published windows is attributed to the first gate "
            "that turned the rule's candidate down (`rejected_events` of the "
            "detector), or to the train test when every gate accepted it, or to "
            "no candidate when no two-channel edge cluster exists there:\n"
        ),
        "\n".join(lines) + "\n",
    ]


def reference_lines(check):
    """Narrative for the frozen rule on the Muscatello reference shots."""
    ratios, central = [], []
    for row in check["by_shot"]:
        for window in row["calibration"].values():
            ratios.append(window["ecevs17_over_ecevs15"])
            central.append(window["central_channel_median_te_kev"])
    observable = {
        row["shot"]: row["frozen_rule"]["states"]["observable_s"]
        for row in check["by_shot"]
    }
    record = check["by_shot"][0]["frozen_rule"]["states"]["record_s"]
    off = {
        row["shot"]: sum(
            w["crashes_found"] for w in row["rule_validity_test_off"]["windows"]
        )
        for row in check["by_shot"]
    }
    rule_found = {
        row["shot"]: len(row["frozen_rule"]["crash_times_s"])
        for row in check["by_shot"]
    }
    unconfirmed = sum(
        len(w["agreement_with_picker"]["unconfirmed_picks_s"])
        for row in check["by_shot"]
        for key in ("frozen_rule", "rule_validity_test_off")
        for w in row[key]["windows"]
    )
    known = check["by_shot"][0]["frozen_rule"].get("known_crash_offset_ms")
    return [
        (
            "The frozen rule and the independent derivative picker were run on "
            "the two published DIII-D reference shots, 141182 and 141195 "
            "(central-channel windows 2.7–3.0 s and 4.55–4.8 s; bands "
            "85 ± 5 ms and 0.35 ± 0.02). Nothing here selects a threshold. "
            "Results by method:\n"
        ),
        rule_reference_table(check),
        (
            f"**The frozen rule fails this check.** With the ECE-validity test "
            f"it has {number(observable[141182], 2)} s of {record:.1f} s "
            f"observable on 141182 and {number(observable[141195], 2)} s on "
            f"141195, and finds {rule_found[141182]} and {rule_found[141195]} "
            "crashes, none of them inside the published windows (both windows "
            "expect 2–4). Picks are not wrong, they are absent: no picks the "
            "picker does not confirm. The cause is the validity step test. "
            f"ECEVS17 reads {min(ratios):.2f}–{max(ratios):.2f} times ECEVS15 "
            "in every window, a static multiplicative calibration step (the "
            "TECEF calibration, applied between ECEVS15 and ECEVS17) that is "
            "constant through the crashes; the central channel reads "
            f"{min(central):.1f}–{max(central):.1f} keV against the published "
            "2–5 keV. The rule treats such an array as an unresolved Te "
            "profile and abstains, which is the intended behaviour for an "
            "uncalibrated array and the cost of that behaviour on these shots.\n"
        ),
        (
            "With only the validity test switched off, as a diagnostic and not "
            f"a rule, the same detector finds {off[141182]} crashes in the "
            f"windows on 141182 and {off[141195]} on 141195, with "
            f"{unconfirmed} picks the picker does not confirm"
            + (
                f" and a pick {known:.2f} ms from the published 2.8371 s crash"
                if known is not None
                else ""
            )
            + ". Its period and amplitude pass or fail the bands as the table "
            "shows. The derivative picker fails some bands too (92.1 ms against "
            "85 ± 5 on 141182, amplitude 0.380 above 0.35 ± 0.02), so the "
            "amplitude definition on the stepped calibration, with the central "
            "channel at about 9 keV, is not comparable with the published 2–5 keV "
            "central Te and these band failures are not evidence about the rule "
            "alone.\n"
        ),
        *reference_gate_lines(check),
        source("muscatello_rule_check.json"),
    ]


def elm_verdict(entry, reference):
    """The 190637 sentence, decided by the D-alpha coincidence record."""
    data = entry["dalpha"]
    if data["status"] != "evaluated":
        return "no filterscope D-alpha, so the span stays indeterminate."
    fraction, chance = data["coincident_fraction"], data["chance_fraction"]
    ref = reference["dalpha"]
    context = (
        f" (reference sawtooth shot {reference['shot']}: "
        f"{number(ref['coincident_fraction'], 2)})"
        if ref.get("status") == "evaluated"
        else ""
    )
    if data["marks_elms"]:
        return (
            f"the expert span marks edge-localized modes, not a central sawtooth. "
            f"{100 * fraction:.0f}% of its {data['events_evaluated']} events lie "
            f"within ±1 ms of a filterscope D-alpha spike, against "
            f"{100 * chance:.0f}% at the same times shifted by 35–100 ms{context}."
        )
    return (
        "not a central sawtooth by ECE, and the D-alpha test does not mark "
        f"edge-localized modes either ({100 * fraction:.0f}% of "
        f"{data['events_evaluated']} events coincide with a spike, against "
        f"{100 * chance:.0f}% shifted{context}), so the span stays indeterminate."
    )


def radial_lines(radial):
    """Radial relative-drop profiles of the reviewed shots and what they show."""
    reference = radial["reference"]
    shots = radial["shots"]
    lines = [
        (
            "Each profile is the median over relaxation events of every ECE "
            "channel's relative Te change across the event (medians over "
            "0.3–1.5 ms after against 0.3–1.5 ms before), plotted against "
            "signed nominal ρ (negative on the high-field side). Resolution is "
            "one ECE channel, about 0.03–0.05 in ρ near the axis, so it cannot "
            "separate structure inside about 0.05 of the axis, and ρ is "
            "geometric, not flux. Events are the rule's accepted crash points "
            "and periodic core edges inside the expert-positive spans (all "
            "events of 186532, which has no reviewed span). The verified "
            f"sawtooth shot {reference['shot']} ({reference['events']} events) "
            f"is the reference: central {number(reference['central_median_change'])}, "
            f"outer rise {number(reference['outer_rise_max_change_rho_0p3_0p6'])}. "
            "The verdict rule was written before any reviewed shot was read: "
            "central drop ≤ −0.05 inside |ρ|<0.15 and an outer rise ≥ +0.02 at "
            "ρ 0.3–0.6 is sawtooth-like; no central drop with a monotone outward "
            "decline to ≤ −0.15 at the outermost channel is edge-driven; "
            "otherwise indeterminate. A second, independent test counts the "
            "fraction of events within ±1 ms of a filterscope D-alpha spike "
            "(6 robust standard deviations over the surrounding ±25 ms), against "
            "the same fraction at the event times shifted by 35–100 ms; a shot "
            "marks edge-localized modes when at least half of its events "
            "coincide with a spike and at least twice the shifted fraction "
            "does.\n"
        ),
        (
            "| Shot | Events | Central change | Outer rise (ρ 0.3–0.6) | "
            "Outermost change (ρ) | Span s | Observable after guards s | "
            "D-alpha coincident / shifted | Verdict |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    def coincidence(entry):
        data = entry["dalpha"]
        if data["status"] != "evaluated":
            return "—"
        return (
            f"{number(data['coincident_fraction'], 2)} / "
            f"{number(data['chance_fraction'], 2)}"
        )

    lines.append(
        f"| {reference['shot']} (reference) | {reference['events']} | "
        f"{number(reference['central_median_change'])} | "
        f"{number(reference['outer_rise_max_change_rho_0p3_0p6'])} | — | — | — | "
        f"{coincidence(reference)} | {reference['verdict']} |"
    )
    for shot, row in shots.items():
        support = row["support"]
        outer = row["outermost_median_change"]
        lines.append(
            f"| {shot} | {row['events']} | {number(row['central_median_change'])} | "
            f"{number(row['outer_rise_max_change_rho_0p3_0p6'])} | "
            f"{number(outer)} ({number(row['outermost_rho'], 2)}) | "
            f"{number(support['expert_positive_s'], 2)} | "
            f"{number(support['observable_s'], 2)} | {coincidence(row)} | "
            f"{row['verdict']} |"
        )
    lines.append("")
    a, b, c, d = (shots[k] for k in ("190637", "186636", "189324", "186532"))
    guards = b["guard_accounting_samples"]
    core = guards["core_observable_samples"]
    lines += [
        (
            f"**Shot 190637: {elm_verdict(a, reference)}** The span is "
            f"{number(a['support']['expert_positive_s'], 2)} s and "
            f"{number(a['support']['observable_s'], 2)} s of it is observable "
            "after the guards. The median event has no central drop "
            f"({number(a['central_median_change'])}, against "
            f"{number(reference['central_median_change'])} for the verified "
            f"sawtooth) and no outer rise ({number(a['outer_rise_max_change_rho_0p3_0p6'])}); "
            "the drop grows outward to "
            f"{number(a['outermost_median_change'])} at ρ≈{number(a['outermost_rho'], 2)}. "
            "The pre-registered edge-driven bar (−0.15) is "
            + (
                f"missed by {abs(-0.15 - a['outermost_median_change']):.2f}, so the "
                "formal ECE verdict is indeterminate"
                if a["verdict"] != "edge-driven"
                else "met, so the formal ECE verdict is edge-driven"
            )
            + ". "
            f"Q-prior time (EFIT01 q_min at least 1.5 for 50 ms) covers "
            f"{number(a['state_seconds']['q_prior_ece_contradicted'] + a['state_seconds']['q_prior_untested'], 2)} s "
            "of the shot, consistent with no q=1 surface. The rule is right "
            "not to call this span a present sawtooth, and the expert-positive "
            "label, taken as a central sawtooth, is not supported by ECE.\n"
        ),
        (
            f"**Shot 186636: cutoff, so the span is untestable, not wrong.** "
            f"Only {number(b['support']['observable_s'], 2)} s of the "
            f"{number(b['support']['expert_positive_s'], 2)} s span "
            f"({100 * b['support']['observable_fraction']:.0f}%) stays observable. "
            "With the local-field density guard the cutoff removes "
            f"{guards['removed_by_density_guard']:,} of {core:,} core-observable samples "
            f"({100 * guards['removed_by_density_guard'] / core:.0f}%; the earlier "
            f"B0(R0) guard removed {100 * guards['removed_by_reference_density_guard'] / core:.0f}%) and "
            f"ECE validity {100 * guards['removed_by_ece_validity'] / core:.0f}% more. "
            "The pre-event Te profile (figure) has a steep edge, falling from "
            "about 2.6 to 1.2 keV within about 5 cm just outside the axis. On the "
            f"events that survive the central change is {number(b['central_median_change'])} "
            f"with an outer rise of {number(b['outer_rise_max_change_rho_0p3_0p6'])}: "
            "the right sign but about a tenth of the reference amplitude, below "
            "the pre-registered thresholds. ECE cannot confirm or refute the "
            "expert span on most of it; the rule abstains correctly "
            f"({number(b['state_seconds']['uncertain'], 2)} s uncertain, "
            f"{number(b['state_seconds']['unassessed'], 2)} s unassessed, no present) "
            "and the expert span stays untested.\n"
        ),
        (
            f"**Shot 189324 agrees.** Central {number(c['central_median_change'])} "
            f"and outer rise {number(c['outer_rise_max_change_rho_0p3_0p6'])} "
            "match the reference sawtooth shape, and the rule labels "
            f"{number(c['state_seconds']['present'], 2)} s present. "
            f"**Shot 186532** (not a reviewed shot; it carried the weak gallery "
            f"crash panel of earlier rounds) shows no "
            f"relaxation at its {d['events']} rule events (central "
            f"{number(d['central_median_change'])}, outer {number(d['outer_rise_max_change_rho_0p3_0p6'])}); "
            "its three accepted crash points are weak and uncertain. The panel "
            "is left as it is: the rule is not changed for one shot and the "
            "weak panel is reported, not replaced.\n"
        ),
    ]
    return lines


def report(args):
    def read(name):
        return json.loads((args.output / name).read_text())

    data = read("data_summary.json")
    cohort, population = read("cohort_labels.json"), read("population_labels.json")
    bench, validation = read("benchmark.json"), read("validation.json")
    q1 = read("q1_radius_audit.json")
    side = q1["paired_point_side"]
    frequency = read("fetched_frequency_audit.json")
    bias, null = read("qmin_bias_audit.json"), read("phase_null_audit.json")
    legacy, queue = read("legacy_disagreement.json"), read("crash_time_queue.json")
    check, manifest = read("muscatello_rule_check.json"), read("label_manifest.json")
    verification = read("verification.json")
    composition = read("absent_composition.json")
    gaps, radial = read("uncertain_gaps.json"), read("radial_drop_profiles.json")
    rule = json.loads((args.output / "freeze.json").read_text())["rule"]
    derivation = read("edge_context_derivation.json")
    models = bench["Tokamak-SI"]
    oof_coverage = models["saw-ours"]["coverage"]
    val_coverage = models["saw-ours"]["fixed_validation"]["coverage"]
    counts = models["saw-ours"]["assessment_totals"]
    percentage = 100 * counts["assessed_bins"] / counts["observable_bins"]
    population_summary = data["population"]
    cohort_before = sum(
        composition["before"][k]["absent_s"] for k in ("train", "val", "test")
    )
    cohort_tested = sum(
        composition["after"][k]["tested_absent_s"] for k in ("train", "val", "test")
    )
    cohort_q_prior = sum(
        composition["after"][k]["q_prior_only_s"] for k in ("train", "val", "test")
    )
    cohort_contradicted = sum(
        composition["after"][k]["q_prior_ece_contradicted_s"]
        for k in ("train", "val", "test")
    )
    cohort_high_q = sum(
        composition["before"][k]["sustained_high_q_absence_s"]
        for k in ("train", "val", "test")
    )
    cohort_observable = sum(
        data["splits"][k]["observable_seconds"] for k in ("train", "val", "test")
    )
    if population_summary["complete"]:
        population_text = (
            "The population run is complete: all "
            f"{population_summary['corpus_shots']} corpus shots were attempted; "
            f"{population_summary['processed_count']} have a usable label record "
            f"and {population_summary['excluded_records']} do not (reader or "
            "physical-core failures; the counts by reason are under Label states)."
        )
    else:
        population_text = (
            "**The population run is not complete under the final rule**: "
            f"{population_summary['processed_count']} of "
            f"{population_summary['corpus_shots']} corpus shots had a record when "
            "this report was written. Population figures below describe that "
            "partial set only and are not a population estimate; no earlier-round "
            "population number is current."
        )
    lines = [
        "# Sawtooth physics-rule labels: current state\n",
        (
            "These are **physics-rule labels validated only by the checks described "
            "here**. There are no blind expert crash times. Neither physical label "
            "accuracy nor improvement over the production catalog is established. "
            "The reviewed spans are anchored to old suggestions and were consulted "
            "in previous rule revisions; they are exploratory, not untouched "
            "validation. No model is recommended as latest or stable.\n"
        ),
        (
            f"The cohort has {len(cohort['processed_shots'])}/{cohort['requested_count']} "
            f"successful records and {cohort['crashes']} diagnostic crash candidates. "
            f"OOF model scores assess {counts['assessed_bins']:,}/"
            f"{counts['observable_bins']:,} observable bins ({percentage:.1f}%).\n"
        ),
        source("cohort_labels.json"),
        (
            f"{population_text} "
            f"{population['crashes']} diagnostic points are in the population "
            "records.\n"
        ),
        source("population_labels.json"),
        (
            "**What changed in absence.** The previous round called time absent "
            "when EFIT01 q_min stayed at or above 1.5 for 50 ms, and "
            f"{100 * cohort_high_q / cohort_before:.1f}% of its cohort absent "
            f"seconds ({number(cohort_before, 0)} s) were that high-q rule alone. "
            "q_min≥1.5 is neither necessary nor sufficient for no sawtooth, so it "
            "no longer makes a negative. Absent now means an ECE quiet-core test "
            f"passed: {number(cohort_tested, 0)} s on the cohort "
            f"({100 * cohort_tested / cohort_observable:.1f}% of observable time). "
            f"Time with sustained q_min≥1.5 and no tested absence is the q-prior "
            f"({number(cohort_q_prior, 0)} s, "
            f"{100 * cohort_q_prior / cohort_observable:.1f}% of observable "
            "time), exported as `uncertain` and excluded from benchmark "
            f"negatives. {100 * cohort_contradicted / cohort_q_prior:.1f}% of it "
            f"({number(cohort_contradicted, 0)} s) lies within the absence-test "
            "context of a periodic edge or a profile candidate "
            "(`q_prior_ece_contradicted`): the ECE shows relaxation evidence "
            "there, so this share measures how much of the old q-only negative "
            "class the ECE contradicts. The rest (`q_prior_untested`) was not "
            "tested by the ECE. The assessed set is therefore smaller and more "
            "positive-heavy; the old-negatives sensitivity below keeps the "
            "previous scoring for comparison.\n"
        ),
        source("absent_composition.json"),
        (
            "Conditional OOF crash F1 at ±2 ms: derivative "
            f"{number(models['saw-derivative']['crash_tolerance_2ms']['crash']['f1'])}, "
            "HL-3-gated derivative "
            f"{number(models['saw-hl3']['crash_tolerance_2ms']['crash']['f1'])}, "
            "saw-ours "
            f"{number(models['saw-ours']['crash_tolerance_2ms']['crash']['f1'])}. "
            f"{best_crash_sentence(models)} "
            "HL-3's expert-shot ranking remains inverted on shot 190637; "
            "independent physical validation remains pending.\n"
        ),
        source("benchmark.json", "Tokamak-SI.*"),
        "### Method\n",
        *method_lines(rule, derivation),
        source("freeze.json", "rule"),
        "### Tested absence and the density and cutoff guards\n",
        (
            "Absent class before (previous round) and after (tested absence), "
            "in seconds of 10 kHz samples:\n"
        ),
        composition_lines(composition),
        (
            "The tested absent seconds split by whether EFIT01 q_min also stayed "
            "high: "
            f"{number(sum(composition['after'][k]['tested_absent_with_high_q_s'] for k in ('train', 'val', 'test')), 0)} s "
            "with high q and "
            f"{number(sum(composition['after'][k]['tested_absent_without_high_q_s'] for k in ('train', 'val', 'test')), 0)} s "
            "without.\n"
        ),
        source("absent_composition.json", "before; after"),
        (
            "Sensitivity to the isolated-edge veto length (5.15 ms is the frame "
            "holdoff; 375 ms is the full absence-test context). The rule's value "
            "was derived on TRAIN shots only; the rebuilt negatives on every "
            "split, and the scored OOF negatives of the PhaseNet-style picker, "
            "follow from the stored frame-holdoff masks without any refit:\n"
        ),
        context_lines(composition, models),
        source(
            "absent_composition.json",
            "edge_context_sensitivity; benchmark.json "
            "Tokamak-SI.saw-ours.edge_context_sensitivity; "
            "edge_context_derivation.json",
        ),
        (
            "The density guard uses the local field at the axis resonance "
            "(F/R_axis) instead of the field at R0 wherever it is mapped, and an "
            "ECE-side validity test removes time where a static channel-to-channel "
            "calibration step or a cold axis channel shows the ECE core is not "
            "resolved; dead channels are left out of that test. "
            "Core-observable seconds and what each guard removed:\n"
        ),
        guard_lines(composition),
        source("absent_composition.json", "guards"),
        (
            "Records by the density-guard branch they took. The cutoff uses the "
            "local axis field wherever it is mapped; the other branches are the "
            "fallbacks:\n"
        ),
        guard_status_lines(composition),
        source("absent_composition.json", "guards.*.records_by_density_guard_status"),
        (
            "The guard accounting for each reviewed shot is in the radial-drop "
            "section below.\n"
        ),
        "### Geometry and equilibrium\n",
        (
            "Channels 0–39 use the archived fixed RF grid; same-shot setup takes "
            "precedence, including the documented exceptional archived shot. "
            "Channels 40–47 are excluded from core, outer, coincidence, "
            "redistribution and inversion evidence. Nominal vacuum resonance is "
            "R=2×27.992 GHz/T×|F_boundary|/f. The EFIT magnetic axis selects the core. "
            "Every time sample with R₂<(2/3)R_LCFS,out is excluded; the shot core "
            "screen also rejects channels in that overlap region. Missing radial "
            "metadata does not establish definite-positive spatial evidence. "
            "Previously terminal-dependent candidates are retained as uncertainty.\n"
        ),
        (
            f"The archived grid audit covers "
            f"{frequency['archived_grid_proof']['shots_audited']} shots with "
            f"{len(frequency['archived_grid_proof']['exceptions'])} documented "
            "setup exception. The fetched cohort setup check matches "
            f"{frequency['matches']}/{frequency['cached_setup_records']} "
            "first-40 grids. The fixed grid is transferred to other population "
            "shots; an unaudited historical setup change cannot be excluded.\n"
        ),
        source("fetched_frequency_audit.json"),
        (
            "The adapted HL-3 outer input uses low-field-side nominal geometric "
            "ρ=0.4–0.65, beyond the typical inversion region, rather than adjacent "
            "array rows. Geometric ρ=|R−R_axis|/(R_LCFS,out−R_axis) is **not** "
            "normalized flux. Vacuum mapping omits relativistic and optical-depth "
            "corrections. Missing ECEZH uses the published first-40 midplane "
            "assumption explicitly. Full EFIT profiles are needed for q=1; "
            "minimal field/axis/boundary metadata cannot supply that comparison.\n"
        ),
        source("geometry_metadata_audit.json"),
        (
            f"The q=1 audit contains {q1['efit_shots']} EFIT-supported shots, "
            f"{q1['q1_checked_shots']} with a profile intersection check, and "
            f"{q1['paired_shots']} with paired diagnostic inversion points. "
            "All-point ΔR (inversion R minus same-side q=1 R, metres): "
            f"{distribution(q1['distribution_all_diagnostic_points'], ' m')}. "
            "The full per-shot ledger includes checks with no axis-connected "
            "surface, checks with no candidate, and missing-profile cases. "
            "Paired nominal differences greater than 0.15 m flag uncertainty; "
            "a missing EFIT01 surface near q≈1 cannot distinguish reconstruction "
            "bias from the observed ECE train. It is recorded as incomparable. "
            f"The comparison is low-field-side dominated: {side['low_field_side_points']} "
            f"of {side['paired_points_with_axis_R']} paired points "
            f"({100 * side['low_field_side_fraction']:.1f}%) lie on the "
            f"low-field side and {side['high_field_side_points']} on the "
            "high-field side, so the high-field side is effectively untested. "
            "The inversion lies close to one radius: |R_inversion − R_axis| is "
            f"{distribution(side['absolute_inversion_R_minus_axis_R_m'], ' m')}.\n"
        ),
        source("q1_radius_audit.json", "paired_point_side"),
        (
            "TRAIN definite-present nominal inversion ρ: "
            f"{distribution(q1['train_present_inversion_nominal_rho'])}. "
            "The shot-median "
            "outer input lies beyond the candidate inversion at "
            f"{100 * q1['train_present_outer_beyond_inversion_fraction']:.1f}% "
            "of comparable TRAIN definite-present points; the full ledger "
            "retains the distribution rather than assuming this holds for "
            "every event.\n"
        ),
        (
            "EFIT01 conflict is q_min>1.4. Sustained q_min≥1.5 for at least 50 ms "
            "is a prior, not an absence test: it marks time "
            "`q_prior_ece_contradicted` (inside the absence-test context of a "
            "periodic edge or profile candidate) or `q_prior_untested`, exported "
            "uncertain and never a benchmark negative. A "
            "magnetics-only reconstruction is not an "
            "MSE-constrained central-current measurement: the review-prescribed "
            "1.3–1.5 band replaces the unsupported 1.05 cutoff, rather than "
            "estimating a calibrated q correction. Only prior TRAIN candidates "
            "enter the bias audit, read from the hashed snapshot of the earlier "
            "round's inputs (`labels/prior_inputs/fix2_inputs.json`). "
            "MSE-constrained q retains the stricter conflict "
            "test. Conflicting inversion-qualified trains remain uncertain over "
            "their full context; isolated POSR edges protect finite edge support "
            "without vetoing an entire high-q phase.\n"
        ),
        (
            "Prior TRAIN candidate q_min: "
            f"{distribution(bias['prior_train_candidate_qmin'])}. "
            "Shot 186532 current state seconds: "
            f"{seconds_text(bias['shot_186532']['state_seconds'])}.\n"
        ),
        source("qmin_bias_audit.json"),
        source("freeze.json"),
        (
            "This TRAIN subset measures sensitivity to the previous cutoff, "
            "not the true EFIT bias. The 1.4/1.5 guards are prescribed "
            "conservative tolerances, with no calibrated q correction. "
            "Muscatello's radial reference uses MSE-constrained EFIT; the "
            "present nominal comparison uses EFIT01. See "
            "[Muscatello et al. (2012)]"
            "(https://doi.org/10.1088/0741-3335/54/2/025006) and the locally "
            "archived `Muscatello_ST.md` digest.\n"
        ),
        "### Relaxation phases and support\n",
        (
            "Phase edges require both a ≥2% fractional drop and POSR≥6. The phase "
            "period floor is 10 ms, above the old 5.15 ms picker holdoff artifact "
            "and conservatively below Muscatello's DIII-D reference periods. "
            "Positive trains retain the frozen 20–250 ms bounds. Phase expansion "
            "requires at least six edges and shuffled-time p≤0.05. Each null "
            "preserves count, span and picker holdoff, repeats the same grouping, "
            "and compares the minimum gap CV over all groups. It accounts for "
            "search and multiplicity within each observable run. Independent "
            "generated noise and regular-train checks test this calibration, not "
            "physical label validity.\n"
        ),
        (
            "In 1,000 independent shuffled trials, the full search accepts "
            f"{null['full_group_search_null']['rejected']}/"
            f"{null['full_group_search_null']['draws']} noise sequences "
            f"({100 * null['full_group_search_null']['rate']:.1f}%; 95% CI "
            f"{100 * null['full_group_search_null']['ci95'][0]:.2f}–"
            f"{100 * null['full_group_search_null']['ci95'][1]:.2f}%). "
            f"It accepts {null['periodic_contrast']['rejected']}/"
            f"{null['periodic_contrast']['draws']} jittered regular trains "
            "and rejects a regular 6 ms sequence. Qualification by POSR is "
            "conditioned on in this timing-null audit, rather than simulated. "
            "There is no global familywise guarantee across shots/runs.\n"
        ),
        source("phase_null_audit.json"),
        "### Label states\n",
        (
            "State seconds. `Absent` is tested absence; the two q-prior states are "
            "q-prior only (exported as uncertain, no benchmark negatives).\n"
        ),
        "| Split | Shots | Present s | Absent s | Q-prior, ECE-contradicted s | Q-prior, untested s | Uncertain s | Unassessed s | Assessed / observable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "val", "test"):
        row = data["splits"][split]
        seconds = row["state_seconds"]
        lines.append(
            f"| {split} | {row['shots_requested']} | "
            + " | ".join(number(seconds[s]) for s in STATES)
            + f" | {100 * row['assessed_fraction_of_observable']:.1f}% |"
        )
    seconds = population_summary["state_seconds"]
    requested = (
        population_summary["processed_count"] + population_summary["excluded_records"]
    )
    lines.append(
        f"| population | {requested} | "
        + " | ".join(number(seconds[s]) for s in STATES)
        + f" | {100 * population_summary['assessed_fraction_of_observable']:.1f}% |"
    )
    messages = ", ".join(
        f"{count} × {message}"
        for message, count in sorted(
            population_summary["exclusion_messages"].items(), key=lambda kv: -kv[1]
        )
    )
    lines += [
        "",
        (
            f"Population: {requested} of {population_summary['corpus_shots']} "
            "corpus shots have a record "
            + (
                "(complete)"
                if population_summary["complete"]
                else "(PARTIAL: the rest are unlabelled and the rows above are not "
                "a population estimate)"
            )
            + f"; {population_summary['processed_count']} are usable and "
            f"{population_summary['excluded_records']} "
            "have none and contribute no state seconds, so their absence is not "
            "evidence of absence. Exclusion reasons: "
            f"{messages}. The population row counts the cohort shots as well. "
            "Records whose archived equilibrium lacks the field or axis metadata "
            "(radius status `bt_or_efit_axis_unavailable`) carry no present "
            "candidates and no absent seconds; they are almost entirely "
            "unassessed."
        ),
        "",
        source("data_summary.json", "splits; population"),
        "### Conditional agreement with the physics rule on assessed bins\n",
        (
            f"OOF shots: {oof_coverage['requested_shots']} requested, "
            f"{oof_coverage['supported_shots']} with usable physical inputs, "
            f"{oof_coverage['scored_shots']} with assessed outcomes. Unsupported "
            "shots remain explicit unknown entries and supply no invented "
            "predictions or negatives.\n"
        ),
        (
            "OOF: fixed TRAIN shots, three whole-shot folds; inner shots select "
            "weights, LR/regularisation and operating points. Fixed val/test "
            "shots are excluded from all selection. Assessment masks affect "
            "loss and scoring, never the input definition. Uncertain and "
            "unassessed bins supply no negative truth. Excluded picks have "
            "unknown outcomes, not established false positives. CIs and "
            "paired comparisons use 1,000 whole-shot bootstrap draws.\n"
        ),
        (
            "The trivial picker differentiates only the EFIT-axis-selected "
            "single ECE channel. Inner shots independently select z for crash "
            "F1 and for the presence rule: any edge ≥zσ within ±125 ms. "
            "Thresholds, fitting/inner/scoring shot IDs and support counts "
            "are recorded for every fold. Always present supplies no crash time.\n"
        ),
        *benchmark_framing(models),
        model_table(models),
        source("benchmark.json", "Tokamak-SI.*.crash_tolerance_2ms; assessment_totals"),
        (
            f"Second held-out set: {val_coverage['requested_shots']} nonexpert "
            f"fixed-validation shots requested; {val_coverage['supported_shots']} "
            f"have usable physical inputs and {val_coverage['scored_shots']} "
            "have assessed outcomes. Frozen "
            "three-fold ensembles and means of inner-selected thresholds. "
            "These still measure conditional teacher agreement.\n"
        ),
        model_table(models, fixed=True),
        source("benchmark.json", "Tokamak-SI.*.fixed_validation"),
        "| Held-out set | Method minus derivative-only | Δ crash F1 [95% CI] | Δ presence F1 [95% CI] |",
        "|---|---|---:|---:|",
    ]
    for fixed, label in ((False, "OOF"), (True, "Fixed validation")):
        for name in ("saw-derivative", "saw-hl3", "saw-ours", "saw-always-present"):
            row = models[name]["fixed_validation"] if fixed else models[name]
            if name == "saw-derivative":
                lines.append(f"| {label} | {NAMES[name]} (reference) | 0 | 0 |")
                continue
            paired = row["paired_vs_derivative"]["crash_tolerance_2ms"]
            lines.append(
                f"| {label} | {NAMES[name]} | "
                f"{interval(paired['difference']['crash_f1'], paired['ci95']['crash_f1'])} | "
                f"{interval(paired['difference']['presence_f1'], paired['ci95']['presence_f1'])} |"
            )
    lines += [
        "",
        source("benchmark.json", "paired_vs_derivative"),
        "#### Sensitivity to the previous negatives\n",
        (
            "Benchmark negatives now come only from ECE-tested absence. The "
            "previous negatives were q-prior time: this table restores "
            "the q-prior states as scoring-only assessed time with no crashes, "
            "using the same fitted models and thresholds. Models are not refit. "
            "A large gap shows how much of the earlier score came from "
            "q-prior time.\n"
        ),
        sensitivity_lines(models),
        source("benchmark.json", "Tokamak-SI.*.old_negatives_sensitivity"),
        "#### Operating points and their spread across folds\n",
        (
            "Presence and crash thresholds are selected on each fold's inner "
            "shots. A wide presence-threshold range for saw-ours means its "
            "presence operating point is not stable across folds, so the pooled "
            "presence F1 mixes operating points. Per fold, presence at one fixed "
            "threshold of 0.5 beside the inner-selected one:\n"
        ),
        operating_table(models),
        source("benchmark.json", "Tokamak-SI.*.operating_points"),
        fixed_threshold_table(models),
        source("benchmark.json", "Tokamak-SI.*.presence_fixed_threshold"),
        (
            "HL-3 is an adapted external architecture, replacing the paper's "
            "SXR pair with geometry-selected ECE plus Mirnov and Ip. It has "
            "no crash head: its timing score is the **derivative picker gated "
            "by HL-3**. LR, weight decay and dropout are selected on inner "
            "shots. The U-Net probability grid extends through 0.999; fold "
            "records retain selected operating points and boundary flags.\n"
        ),
        source("saw-hl3_fold_0.json"),
        source("saw-hl3_fold_1.json"),
        source("saw-hl3_fold_2.json"),
        source("saw-ours_fold_0.json"),
        source("saw-ours_fold_1.json"),
        source("saw-ours_fold_2.json"),
        "| Three-regime window classifier | Accuracy | Macro-F1 |",
        "|---|---:|---:|",
    ]
    classification = models["saw-hl3"]["three_class"]
    rows = (
        ("Adapted HL-3", classification),
        (
            "Single-channel derivative with period",
            models["saw-derivative"]["three_class"],
        ),
        ("Fitting-chosen majority", classification["majority_baseline"]),
    )
    for name, row in rows:
        lines.append(
            f"| {name} | {number(row['window_accuracy'])} | {number(row['macro_f1'])} |"
        )
    lines += [
        "| Always present (period class undefined) | — | — |",
        "",
        (
            "Derivative period abstentions count as errors on the original "
            "class support. Always present has no period class or crash time.\n"
        ),
        source("benchmark.json", "Tokamak-SI.*.three_class"),
        (
            "Published HL-3 context, different task/population: real-time "
            f"accuracy stated {number(bench['legacy']['real_time']['accuracy_stated'])}, "
            "count-derived "
            f"{number(bench['legacy']['real_time']['accuracy_from_counts'])}; "
            "offline accuracy stated "
            f"{number(bench['legacy']['offline']['accuracy_stated'])}, count-derived "
            f"{number(bench['legacy']['offline']['accuracy_from_counts'])}. "
            "These three-regime classification scores are not DIII-D crash "
            "scores. The stated and count-derived accuracies differ in the "
            "source.\n"
        ),
        source("benchmark.json", "legacy"),
        "### Exploratory reviewed spans and old-rule disagreement\n",
        (
            "| Method | Pooled reviewed-span AUROC | Per-shot reviewed-span F1 | "
            "Known / excluded / observable picks |"
        ),
        "|---|---:|---|---|",
    ]
    for name, label in NAMES.items():
        row = models[name]["expert"]
        cells = "; ".join(
            f"{r['shot']}: {number(r['presence']['f1'])}" for r in row["by_shot"]
        )
        picks = row["pick_totals"]
        lines.append(
            f"| {label} | {number(row['presence']['auroc'])} | {cells} | "
            f"{picks['assessed_picks']} / {picks['excluded_picks']} / "
            f"{picks['observable_picks']} |"
        )
    lines += [
        "",
        source("benchmark.json", "Tokamak-SI.*.expert"),
        (
            "Reviewed-score support is the intersection of known reviewed spans "
            "and observable inputs. Picks outside it have unknown outcomes. "
            "The JSON separately records picks excluded by physics-rule "
            "assessment for the conditional agreement diagnostic. Model "
            "predictions on reviewed spans are scored independently of the "
            "physics rule's assessment mask.\n"
        ),
        (
            f"HL-3 ranking check: {models['saw-hl3']['expert']['auc_diagnosis']}. "
            "No expert-based inversion or retuning was performed.\n"
        ),
        "| Reviewed shot | HL-3 AUROC | HL-3 assessed diagnostic AUROC | Saw-ours AUROC |",
        "|---|---:|---:|---:|",
    ]
    ours_by_shot = {row["shot"]: row for row in models["saw-ours"]["expert"]["by_shot"]}
    for row in models["saw-hl3"]["expert"]["by_shot"]:
        lines.append(
            f"| {row['shot']} | {number(row['presence']['auroc'])} | "
            f"{number(row['conditional_assessed_presence']['auroc'])} | "
            f"{number(ours_by_shot[row['shot']]['presence']['auroc'])} |"
        )
    lines += [
        "",
        source("benchmark.json", "Tokamak-SI.*.expert.by_shot"),
        (
            "HL-3's residual inversion is concentrated in shot 190637 and "
            "persists on assessed support. The physics rule calls no "
            "definite-present phase there while the reviewed spans contain "
            "substantial positive support. The radial profile below resolves "
            "which side the data support.\n"
        ),
        (
            f"Source: `{args.work}/shots/190637.json` → `state_seconds`; "
            "`outputs/labeler/sawtooth/fix5/validation.json` → `expert.by_shot`.\n"
        ),
        "### Radial drop profiles of the reviewed spans\n",
        *radial_lines(radial),
        (
            "Figures: `$LABELER_ROOT/round4/saw/fix5/figures/radial_drop_<shot>.pdf` "
            "and `.png` for 186636, 189324, 190637 and 186532.\n"
        ),
        source("radial_drop_profiles.json"),
        (
            "| Reviewed shot | New recall | Legacy recall | New F1 | "
            "Uncertain / observable positive bins |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for row in validation["expert"]["by_shot"]:
        lines.append(
            f"| {row['shot']} | {number(row['new']['presence']['recall'])} | "
            f"{number(row['old']['presence']['recall'])} | "
            f"{number(row['new']['presence']['f1'])} | "
            f"{row['expert_positive_uncertain_bins']} / "
            f"{row['expert_positive_observable_bins']} |"
        )
    lines += [
        "",
        "Span-supported picks are not crash precision/recall.\n",
        source("validation.json", "expert; legacy_agreement"),
        (
            "Native clipped legacy replay reproduces every retained pick "
            "array on the seven requested shots; clocks are exact. This "
            "excludes a reader timestamp bug. The ledger separates missing "
            "raw edges, greedy 10 ms suppression and inversion-window "
            "rejection; only training shots enter the diagnostic holdoff "
            "ablation. The current catalog detector and the retained legacy "
            "detector are distinct, so a catalog-wide defect cannot be "
            "inferred from the legacy cache alone.\n"
        ),
        source("legacy_reader_audit.json"),
        "Per-shot timing and gate diagnosis (old minus physics, ms):\n",
        timing_table(legacy),
        (
            "Prior offsets refer to the superseded rule, whose spatial "
            "selection included unverified channels. Fresh offsets retain "
            "pairs within 15 ms; nearest neighbours are descriptive and can "
            "be reused. One-to-one cells are separate. Shots 190602/190604 "
            "have too few uncontaminated channels at their field for a "
            "verified redistribution profile and correctly abstain. "
            "The train-only ablation changes holdoff solely for diagnosis. "
            "Its full ledger gives the inversion rejection reasons and "
            "core/outer changes at both rules' picks. Shortening the legacy "
            "profile windows alone does not recover the matches: its block "
            "interiority, gain adjacency and amplitude conditions reject "
            "many central derivative edges.\n"
        ),
        source("legacy_disagreement.json", "by_shot; figures"),
    ]
    current = next(r for r in legacy["by_shot"] if r["shot"] == 192090)
    rows = current["current_production_v3_comparison"]["by_diagnostic"]
    lines += [
        (
            "| Shot 192090 current catalog | Total picks | In physics-present "
            "support | TP / FP / FN ±2 ms | Median offset ms |"
        ),
        "|---|---:|---:|---|---:|",
    ]
    for row in rows:
        cells = " / ".join(number(v, 0) for v in row["present_cells_2ms"])
        lines.append(
            f"| {row['diagnostic']} | {row['current_catalog_count']} | "
            f"{row['physics_present_catalog_count']} | {cells} | "
            f"{number(row['catalog_minus_physics_nearest_ms']['median'], 3)} |"
        )
    lines += [
        "",
        (
            "These TP/FP/FN cells name algorithm agreement against the physics "
            "rule. They do not establish physical errors. The current detector "
            "does not reproduce the retained legacy's systematic 10 ms offset "
            "on this available shot. The other six current event stores are "
            "unavailable locally.\n"
        ),
        source("legacy_disagreement.json", "by_shot.current_production_v3_comparison"),
        "### Uncertain gaps inside present trains\n",
        *gap_lines(gaps),
        "### Muscatello references and blind queue\n",
        *reference_lines(check),
        (
            f"Blind queue: {queue['windows']} windows on {queue['shot_count']} "
            f"shots, including {queue['random_primary_windows']} primary "
            f"random windows. Stratum counts: `{queue['strata']}`. "
            "Random observable windows are frozen "
            "before prediction access. Candidate-free, model-negative, "
            "uncertain and disagreement supplements are separate strata. "
            "Only sensor inputs and blank targets enter the annotation pack; "
            "private selection and all predictions remain hidden. About "
            "97 independent positive events give a worst-case 95% recall "
            "half-width of 0.1; clustered events do not supply that effective "
            "sample size automatically. Primary sampling weights and "
            "whole-shot CIs are preregistered. Annotation remains pending.\n"
        ),
        source("crash_time_queue.json"),
        "### Artifacts, reproduction and verification\n",
        (
            f"Complete labels: `$LABELER_ROOT/round4/saw/fix5/labels/`. "
            f"Manifest `SHA256SUMS` sha256: `{manifest['manifest_sha256']}`. "
            f"{population_text} "
            "Population shards export current states without duplicate cohort "
            "rows; failed-shot records remain in the ledger. The cohort bundle "
            "is separate. Production stores were read only. The manifest also "
            "hashes `prior_inputs/fix2_inputs.json`, the snapshot of the "
            "previous round's inputs the rule reads (terminal-dependent "
            "candidate times, failed-shot list, TRAIN q_min conflict "
            "candidates), so the labels no longer depend on a deleted "
            "directory; `freeze.json` is hashed beside them.\n"
        ),
        source("label_manifest.json"),
        (
            "Large signals, EFIT metadata, models, predictions, annotation "
            "pack, PDFs and 150-dpi PNGs are under the same fix5 directory. "
            "Paper example, three old-rule figures, the 12-shot gallery and "
            "the four radial-profile figures were inspected; figure ledgers "
            "retain paths and hashes.\n"
        ),
        source("paper_example.json"),
        source("gallery.json"),
        source("figure_inspection.json"),
        (
            f"Covering tests: {verification['covering_tests_passed']} passed. "
            "Ruff passes on all changed Python files; formatting passes on "
            "all new Python files; git diff --check passes. Long jobs used "
            "timeouts and logs; temporary storage was swept afterward. "
            "GPU training used CUDA_VISIBLE_DEVICES=1 within the assigned "
            "memory budget.\n"
        ),
        source("verification.json"),
        "```text\n"
        + "\n".join(
            line
            for name in ("covering_tests", "ruff", "format")
            for line in verification["output_tails"][name]
        )
        + "\n```\n",
        (
            "Reproduce with the prescribed pixi environment and TMPDIR. "
            "Entrypoints: sawtooth_edge_context.py run/derive (TRAIN only, "
            "first); sawtooth_fix5_setup.py; "
            "sawtooth_geometry_fix3.py audit/frequency-audit/reference; "
            "sawtooth_physics.py labels (cohort, then population with "
            "sawtooth_population.sbatch); sawtooth_reference_rule.py; "
            "sawtooth_fix3_records.py records/manifest; "
            "sawtooth_absent_composition.py; sawtooth_uncertain_gaps.py; "
            "sawtooth_radial_drop.py; sawtooth_fix_validation.py validate; "
            "sawtooth_fix3_artifacts.py queue-base/reader-audit/queue/figures; "
            "sawtooth_benchmark.py train/predict/evaluate; sawtooth_gallery.py; "
            "sawtooth_phase_null_audit.py; sawtooth_fix3_report.py. "
            "Run queue-base before model prediction access. Full command "
            "logs remain under the large output directory.\n"
        ),
        "### Concerns and next work\n",
        *limitation_lines(
            args.work, data, check, bench, composition, gaps, radial, side
        ),
        (
            "Blind physical accuracy remains unmeasured. Reviewed-span "
            "failures and any inverted HL-3 ranking remain failures, not "
            "reasons to retune on those shots. Nominal geometry is not a "
            "calibrated flux measurement; missing field/profile data limits "
            "population evidence. Algorithm-assessed benchmarks remain "
            "conditional and can be solved by derivative rules. Next: obtain "
            "the queued blind crash/span/ambiguity annotations, lock them, "
            "and evaluate all frozen methods over independently observable "
            "support without retuning.\n"
        ),
        "### Appendix: superseded history\n",
        (
            "Earlier rounds used a hottest-channel proxy, sparse same-shot "
            "RF localization, a q_min>1.05 conflict and unbounded fractional "
            "edge phases. Their results are superseded. The first-round "
            "`outputs/labeler/sawtooth/fix` records and the two earlier plan "
            "documents were removed; `outputs/labeler/sawtooth/fix2` and `fix3` "
            "remain as predecessor records (fix3 counted q_min≥1.5 time as "
            "absent; its absent-class and score numbers are superseded by this "
            "round), and the earlier rounds' narrative is in the stream report "
            "appendix.\n"
        ),
    ]
    body = "\n".join(lines[1:])
    # Repository doc: subsections are level two under the title.
    text = lines[0] + "\n" + re.sub(r"(?m)^### ", "## ", body)
    (REPO / "docs/labeler/sawtooth_results.md").write_text(text)
    (REPO / "docs/labeler/sawtooth_physics.md").write_text(
        "# Sawtooth physics-rule method\n\n"
        "The current method, thresholds, validations, results, limitations and "
        "reproduction commands are in [the current-state report]"
        "(sawtooth_results.md). Labels are physics-rule labels validated only "
        "by the checks reported there; blind expert crash annotations are "
        "pending. Production labels are not replaced.\n"
    )
    if args.report:
        # Appended as the last section; an earlier copy of this section is
        # replaced so reruns stay idempotent.
        marker = "\n## Fix round 4\n"
        existing = args.report.read_text()
        if marker in existing:
            existing = existing[: existing.index(marker)]
        status = (
            f"Status: DONE_WITH_CONCERNS. Commit range: `{args.report_commit_range}`.\n"
        )
        args.report.write_text(
            existing.rstrip("\n") + "\n" + marker + "\n" + status + "\n" + body
        )
    update_readme(manifest, population_summary)


def best_crash_sentence(models):
    """Which method has the best crash-F1 point estimate and the paired gaps."""
    f1 = {
        name: models[name]["crash_tolerance_2ms"]["crash"]["f1"]
        for name in ("saw-derivative", "saw-hl3", "saw-ours")
    }
    best = max(f1, key=f1.get)
    parts = []
    for name in ("saw-hl3", "saw-ours"):
        paired = models[name]["paired_vs_derivative"]["crash_tolerance_2ms"]
        low, high = paired["ci95"]["crash_f1"]
        verdict = "includes zero" if low <= 0 <= high else "excludes zero"
        parts.append(
            f"{NAMES[name]} minus the derivative picker is "
            f"{number(paired['difference']['crash_f1'])} "
            f"[{number(low)}, {number(high)}] ({verdict})"
        )
    return (
        f"{NAMES[best]} has the highest crash-F1 point estimate; the paired "
        f"OOF differences in crash F1 are: {'; '.join(parts)}."
    )


def failure_breakdown(population):
    """Why population shots carry no label, from the exclusion messages."""
    groups = {
        "absent waveform or incompatible clock": 0,
        "insufficient physical or finite ECE core (core channels, finite window, "
        "no ECE group)": 0,
        "unreadable file or object": 0,
        "other (for example a geometry time axis that is not increasing)": 0,
    }
    keys = list(groups)
    for message, count in population["exclusion_messages"].items():
        if "absent waveform" in message:
            groups[keys[0]] += count
        elif any(w in message for w in ("ECE core", "finite ECE", "no ECE group")):
            groups[keys[1]] += count
        elif "Unable to synchronously open" in message:
            groups[keys[2]] += count
        else:
            groups[keys[3]] += count
    return "; ".join(f"{count} × {name}" for name, count in groups.items() if count)


def update_readme(manifest, population):
    path = REPO / "data/events/sawtooth_oscillation/README.md"
    text = path.read_text()
    inputs = text.index("**saw-hl3**:", text.index("## Inputs"))
    method = text.index("## Method\n", inputs)
    # Keep the production ece_sawtooth description; replace everything from
    # the physics-rule paragraphs (first run) or the previous rewrite onward.
    tails = [
        text.find(marker, method)
        for marker in (
            "`labeler.sawtooth.physics.detect`",
            "## Physics-rule labels and validation",
        )
    ]
    tail = min(index for index in tails if index >= 0)
    last = text.index("## Alias", tail)
    inputs_content = """**saw-hl3**:
- EFIT-axis core ECE and low-field-side outer ECE at nominal geometric ρ=0.4–0.65
- Mirnov 0–1 mean and Ip in MA; missing values use fitting-shot means

**saw-ours**:
- The physical first-40-channel ECE array, 100 ms context at 10 kHz; unverified
  channels 40–47 and third-harmonic overlap samples are masked

"""
    population_note = (
        " (cohort and population shards)"
        if population["complete"]
        else " (cohort shards; the population run was not complete when this "
        "was written, so the population shards hold only part of the corpus)"
    )
    population_status = (
        f"The population run is complete: {population['processed_count']} of "
        f"{population['corpus_shots']} corpus shots have a usable record under the "
        f"final rule. The other {population['excluded_records']} carry no label, "
        f"for these reasons: {failure_breakdown(population)}."
        if population["complete"]
        else "The population run is NOT complete under the final rule: "
        f"{population['processed_count']} of {population['corpus_shots']} corpus "
        "shots have a record, and no population number here is current."
    )
    content = f"""## Physics-rule labels and validation

These are **physics-rule labels validated only by the checks described** in the
[current-state report](../../../docs/labeler/sawtooth_results.md). The rule uses
Gude-style POSR, multichannel coincidence, core loss / outer gain, central drop,
stable trains and nominal EFIT localization. It screens harmonic overlap and
uses EFIT01 bias-aware q-min conflicts. POSR-qualified phase edges require
≥10 ms periods and a shuffled-time null that repeats the same group search.
Geometry is nominal, with no flux calibration. Present / absent /
q_prior_ece_contradicted / q_prior_untested / uncertain / unassessed states
remain distinct. Production labels are not replaced. Old `ece_sawtooth` disagreement and its reader audit are in
the report; the retained legacy rule and current catalog detector differ.
Valid core ECE defines observability: missing ECE, low temperature and detected
cutoff yield `unassessed`, and trains split at observability gaps. Native-rate
antialiasing precedes decimation to 10 kHz. Where local neutron-rate and Mirnov
data exist, their drop/burst flags give optional corroboration; no SXR
corroboration is claimed without verified core/edge spatial pairing. **Absent**
means a tested absence: no POSR-periodic core edge on any valid ECE channel at
nominal ρ<0.5 over a complete ±375 ms context, noise-resolved on at least two
channels (an edge counts only when its relative change is at least 2%, so a
core relaxation train below 2% is called quiet). Sustained EFIT01 q-min ≥ 1.5 is
a prior, not a test: that time is `q_prior_ece_contradicted` where it lies in
the absence-test context of a periodic edge or profile candidate (the ECE shows
relaxation evidence there) and `q_prior_untested` elsewhere; both are exported
as `uncertain` with the state name as the reason and are never a benchmark
negative. A density-cutoff guard uses the local field at the axis resonance
where it is mapped (Bt at R0, then a fixed guard, only as fallbacks; the report
gives the records by branch) and an ECE validity test (adjacent-channel step
ratio above 2, or a near-axis channel below 0.6 of the profile maximum,
sustained 20 ms; dead channels are excluded) marks unresolved ECE `unassessed`.
Ambiguous observable support remains uncertain. The
untracked exports in `extend_saw_physics/` hold the spans and crash points; they
are additive research labels.

{population_status}
Population label shards are at
`$LABELER_ROOT/round4/saw/fix5/labels/`{population_note}. Verify with
`sha256sum -c SHA256SUMS` from that directory. The `SHA256SUMS` file has sha256
`{manifest["manifest_sha256"]}`; individual CSV hashes are in
`outputs/labeler/sawtooth/fix5/label_manifest.json`. The manifest also hashes
`prior_inputs/fix2_inputs.json`, a snapshot of the earlier round's inputs the
rule reads, and `freeze.json`. Cohort shards are the `cohort-*.csv` files in the
same directory; `extend_saw_physics/` is the untracked integration copy. Git does
not carry the large label store.

Both learned models use three whole-shot TRAIN folds with inner-shot selection
of checkpoint, hyperparameters and thresholds; CUDA training stops on
inner-selection loss patience. The trivial derivative and always-present
baselines use the same folds. The three reviewed shots and the blind test split
are excluded from training and tuning. `saw-hl3` receives adapted inputs
(EFIT-axis core ECE, low-field-side outer ECE, Mirnov, Ip) while `saw-ours`
receives the first 40 ECE channels, so comparisons include input information as
well as architecture. The second held-out set is the 47 nonexpert
fixed-validation shots. Headline scores are **conditional agreement with the
physics rule on assessed bins** and include excluded-pick counts and paired
shot-bootstrap comparisons. HL-3 crash timing is **derivative picker gated by
HL-3**, an adapted baseline, rather than a learned crash head. Reviewed spans
were anchored to old suggestions and used in previous rule revisions; they
are exploratory and provide no independent crash-time precision/recall; the
190637 span may include edge-originated relaxations.

## Blind crash-time annotation queue

The prediction-free input pack is
`$LABELER_ROOT/round4/saw/fix5/annotation_pack/`. It contains sensor windows,
observable masks, nominal geometry and blank annotation targets. Random windows
are frozen before prediction access. Candidate-free, model-negative, uncertain
and disagreement cases supplement the primary probability sample. Annotators
receive the annotation pack only: no repository outputs and no access to
`round4/saw`. Keep the private `selection_audit/` directory and every
detector/model prediction hidden (the `predictions/` directories are mode
`go-rwx`).
Mark crash times, timing tolerances, positive/negative observable spans and
ambiguity masks; lock annotations before revealing picks. Use preregistered
sampling weights and whole-shot bootstrap intervals. Approximately 97
independent positive events give a worst-case 95% recall half-width of 0.1;
shot clustering reduces the effective count. The owner is away: annotation is
pending and physical accuracy remains unvalidated. No model is recommended.

"""
    text = text[:inputs] + inputs_content + text[method:tail] + content + text[last:]
    text = text.replace(
        "q conflicts abstain)", "nominal geometry and bias-aware q evidence)"
    )
    text = text.replace(
        "sawtooth, sawtooth, sawtooth oscillation", "sawtooth, sawtooth oscillation"
    )
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--report-commit-range", default="ad0ca40f..r4-saw")
    report(parser.parse_args())


if __name__ == "__main__":
    main()
