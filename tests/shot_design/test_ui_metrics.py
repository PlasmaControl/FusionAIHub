"""The UI consumes the Part F6 metrics schema without translating its scores."""

METRICS = {
    "schema": "shot-design-simulation-metrics-v1", "members": 8, "k0": 20,
    "n_predict": 80, "frame_s": 0.05, "t0_s": 2.0, "decode_steps": 10,
    "temperature": 1.0, "held": ["co2"],
    "modalities": {
        "mhr": {
            "family": "spectro", "feature": "10-60 kHz band power",
            "nrmse": {"real": 1.21, "persistence": 1.05, "seed_mean": 1.30},
            "crps": {"real": 0.41, "persistence": 0.38}, "skill": -0.08,
            "spread_error": 0.83, "effect": 0.12, "noise": 0.10,
            "effect_to_noise": 1.2, "resolved": False,
        },
        "co2": {"held": True},
    },
}
