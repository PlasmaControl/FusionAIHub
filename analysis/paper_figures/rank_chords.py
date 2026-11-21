"""Rank slow-TS chords per figure: replay (correlation x signal level) and response (scenario - real)."""
import sys
import numpy as np

D = "/scratch/gpfs/EKOLEMEN/ps9551/ignite/demo_figures_live_unroll/ens4_soup/"
MODS = ("ts_core_temp", "ts_core_density", "cer_ti", "cer_rot")


def live_end(z):
    te = np.asarray(z["gt__real__ts_core_temp"], float)
    core = np.nanmax(te, axis=1)
    on = np.where(core > 0.3 * np.nanmax(core))[0]
    return int(on[-1]) if len(on) else te.shape[0]


def replay(f, k0=20, top=4):
    z = np.load(D + f, allow_pickle=True)
    end = live_end(z)
    out = {}
    for m in MODS:
        g, p = np.asarray(z[f"gt__real__{m}"], float), np.asarray(z[f"pred__real__{m}"], float)
        rows = []
        for c in range(g.shape[1]):
            gg, pp = g[k0:end, c], p[k0:end, c]
            ok = np.isfinite(gg) & np.isfinite(pp)
            if ok.mean() < 0.9 or np.nanstd(gg) == 0:
                continue
            r = np.corrcoef(gg[ok], pp[ok])[0, 1]
            rmse = np.sqrt(np.mean((gg[ok] - pp[ok]) ** 2))
            pers = np.sqrt(np.mean((gg[ok] - g[k0 - 1, c]) ** 2)) if np.isfinite(g[k0 - 1, c]) else np.nan
            rows.append((c, r, rmse / (pers + 1e-9), np.nanmean(np.abs(gg)), np.nanstd(gg)))
        rows.sort(key=lambda x: -(x[1] * (x[2] < 1)) * x[4])
        out[m] = rows[:top]
        print(f, m, "end", end, " ".join(f"ch{c}: r{r:+.2f} rmse/pers {q:.2f} |x| {a:.0f} sd {s:.0f}" for c, r, q, a, s in rows[:top]))
    return out


def response(f, run, mods, k0=50, top=4):
    z = np.load(D + f, allow_pickle=True)
    end = live_end(z)
    w = slice(k0 + 10, end - 2)
    for m in mods:
        r, s = np.asarray(z[f"pred__real__{m}"], float), np.asarray(z[f"pred__{run}__{m}"], float)
        g = np.asarray(z[f"gt__real__{m}"], float)
        d = np.nanmean(s[w] - r[w], 0)
        rel = d / (np.nanstd(g[k0:end], 0) + 1e-9)
        level = np.nanmean(r[w], 0)
        idx = [c for c in np.argsort(-np.abs(d)) if np.isfinite(d[c])][:top]
        print(f, run, m, "end", end, " ".join(f"ch{c}: {level[c]:.0f}->{level[c]+d[c]:.0f} ({rel[c]:+.1f} sd)" for c in idx))


if __name__ == "__main__":
    replay("demo_204011_val_data.npz")
    replay("demo_201081_cm_data.npz")
    replay("demo_201085_cm_data.npz")
    replay("demo_200729_g1_data.npz")
    for run in ("tq_lin_zero", "tq_lin_half"):
        response("ramps/demo_204011_ramps_data.npz", run, ("cer_rot", "cer_ti", "ts_core_temp"))
    for run in ("ech_lin_off", "ech_lin_half"):
        response("ramps/demo_204011_ramps_data.npz", run, ("ts_core_temp", "cer_ti", "ts_core_density"))
        response("ramps/demo_201240_ramps_data.npz", run, ("ts_core_temp", "cer_ti", "ts_core_density"))
