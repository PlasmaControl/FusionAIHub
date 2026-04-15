import pickle
import numpy as np



def compute_beam_avg_power(time, values):
    """
    Compute the requested average beam power from modulated beam measurements.

    Beams are either fully ON or OFF. To achieve a desired average power below the
    peak, the beam modulates (cycles ON/OFF). The duty cycle over each modulation
    period determines the effective requested average power:
        avg_power = peak_power_during_on * (T_on / T_cycle)

    This function detects ON/OFF segments, groups them into modulation cycles,
    and computes the average power per cycle.

    Parameters
    ----------
    time : array-like
        Time array in seconds.
    values : array-like
        Beam power measurements corresponding to each time point.

    Returns
    -------
    np.ndarray
        Requested average beam power at each time point.
    """
    time = np.asarray(time, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)

    # Threshold: beam is ON if power > 1% of the max observed power
    peak = values.max()
    if peak == 0:
        return np.zeros_like(values)
    threshold = 0.01 * peak

    is_on = values > threshold

    # Detect segment boundaries
    changes = np.diff(is_on.astype(int))
    change_idx = np.where(changes != 0)[0] + 1  # index where new state starts

    # Build segments: (start_idx, end_idx, is_on_state)
    boundaries = np.concatenate(([0], change_idx, [len(values)]))
    segments = []
    for i in range(len(boundaries) - 1):
        si, ei = boundaries[i], boundaries[i + 1]
        segments.append((si, ei, bool(is_on[si])))

    # Group into modulation cycles and compute average power per cycle.
    # A cycle = one ON segment + one adjacent OFF segment (in either order).
    # We integrate power over the full cycle: avg = integral / T_cycle.
    avg_power = np.full(len(values), np.nan)

    i = 0
    while i < len(segments):
        si, ei, state = segments[i]
        t_start = time[si]
        t_end = time[ei - 1]

        if not state:
            # OFF segment alone (e.g. before beam starts) -> avg = 0
            avg_power[si:ei] = 0.0
            i += 1
            continue

        # ON segment: look for a following OFF segment to form a cycle
        cycle_start = si
        cycle_end = ei
        if i + 1 < len(segments) and not segments[i + 1][2]:
            cycle_end = segments[i + 1][1]
            consumed = 2
        else:
            consumed = 1

        # Also check if there's a preceding OFF that belongs to this cycle
        # (for the first ON after a long OFF, the OFF already got assigned 0)

        # Compute average power over the cycle using trapezoidal integration
        cycle_time = time[cycle_start:cycle_end]
        cycle_vals = values[cycle_start:cycle_end]
        if len(cycle_time) > 1:
            integral = np.trapezoid(cycle_vals, cycle_time)
            duration = cycle_time[-1] - cycle_time[0]
            cycle_avg = integral / duration if duration > 0 else 0.0
        else:
            cycle_avg = cycle_vals[0]

        avg_power[cycle_start:cycle_end] = cycle_avg
        i += consumed

    # Forward-fill any remaining NaNs
    mask = np.isnan(avg_power)
    if mask.any():
        idx = np.where(~mask, np.arange(len(avg_power)), 0)
        np.maximum.accumulate(idx, out=idx)
        avg_power = avg_power[idx]
        still_nan = np.isnan(avg_power)
        if still_nan.any():
            first_valid = np.argmin(still_nan)
            avg_power[:first_valid] = avg_power[first_valid]
    return avg_power


# ---------- Demo ----------
if __name__ == "__main__":
    data_path = "/scratch/gpfs/aj17/datasets/fm_test/dynamicmodel/dataset/206527_df.pkl"
    with open(data_path, "rb") as fh:
        data = pickle.load(fh)

    nbi = data["nbi"]
    beam_col = "bmspinj33l"
    beam = nbi[beam_col]
    t = beam.index.values.astype(np.float64)
    v = beam.values.astype(np.float64)

    avg_power = compute_beam_avg_power(t, v)

    # Print some diagnostics
    peak = np.median(v[v > 0])
    print(f"Beam: {beam_col}")
    print(f"Peak power (median when ON): {peak:.0f}")
    print(f"Time range: {t[0]:.1f} - {t[-1]:.1f} s")
    print()

    # Show duty cycle at various time points
    is_on = v > 0.01 * v.max()
    segments = []
    changes = np.diff(is_on.astype(int))
    change_idx = np.where(changes != 0)[0] + 1
    boundaries = np.concatenate(([0], change_idx, [len(v)]))
    for i in range(len(boundaries) - 1):
        si, ei = boundaries[i], boundaries[i + 1]
        segments.append((t[si], t[ei - 1], bool(is_on[si])))

    print("Segment analysis:")
    print(f"{'Start':>10s} {'End':>10s} {'State':>5s} {'Duration':>10s} {'Avg Power':>12s} {'Duty Cycle':>12s}")
    for start, end, state in segments[:20]:
        dur = end - start
        label = "ON" if state else "OFF"
        avg_at_start = avg_power[np.argmin(np.abs(t - start))]
        duty = avg_at_start / peak if peak > 0 else 0
        print(f"{start:10.1f} {end:10.1f} {label:>5s} {dur:10.1f}s {avg_at_start:12.0f} {duty:12.1%}")