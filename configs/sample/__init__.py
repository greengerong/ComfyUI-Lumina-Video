CANDIDATE_SAMPLE_CONFIGS = {  # noqa
    "f24F96R960": {
        "P": "1x2",
        "step": 70,
        "ts": 24.0,
        "renorm": 2,
        "cfg": (20, (0.2, 0)),
        "motion": 6.0,
        "negMotion": (2.0, (0.05, 6.0)),
    },
    "f24F96R960-MultiScale": {
        "P": ("2x4", (0.03, "2x2"), (0.125, "1x2")),  # for multi-scale inference
        "step": 70,
        "ts": 24.0,
        "renorm": 2,
        "cfg": (20, (0.2, 0)),
        "motion": 6.0,
        "negMotion": (2.0, (0.05, 6.0)),
    },
    "f16F64R512": {
        "P": "1x2",
        "step": 70,
        "ts": 8.0,
        "renorm": -1,
        "cfg": (12, (0.4, 0)),
        "motion": 6.0,
        "negMotion": (2.0, (0.05, 6.0)),
    },
    "f16F64R512-MultiScale": {
        "P": ("2x4", (0.075, "2x2"), (0.3, "1x2")),  # for multi-scale inference
        "step": 70,
        "ts": 8.0,
        "renorm": -1,
        "cfg": (12, (0.4, 0)),
        "motion": 6.0,
        "negMotion": (2.0, (0.05, 6.0)),
    },
}
