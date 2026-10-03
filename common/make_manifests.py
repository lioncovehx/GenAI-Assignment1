"""Write manifests/val.json and manifests/test.json once (deterministic val/test corruptions).

Run: python -m common.make_manifests
"""
import json

import numpy as np

from common.corruptions import LABELS, SEVERITIES, TEST_LEVELS, make_params, sample_params, severity_bin
from common.data import MANIFESTS, SEED, get_images


def row(image: int, params: dict, severity: str | None, seed: int) -> dict:
    return {"image": image, "name": LABELS[params["label"]], "severity": severity, "rng_seed": seed, **params}


def build_val() -> list:
    """One entry per val image, sampled exactly like training (uniform type, continuous severity)."""
    rows = []
    for i in range(len(get_images("val"))):
        seed = SEED + i
        p = sample_params(np.random.default_rng(seed))
        rows.append(row(i, p, severity_bin(p), seed))
    return rows


def build_test() -> list:
    """Per test image: clean + 3 corruptions x 3 fixed severities = 10 entries."""
    rows = []
    for i in range(len(get_images("test"))):
        seed = SEED * 1_000_000 + i * 10
        rows.append(row(i, make_params(0, np.random.default_rng(seed)), None, seed))
        for label, levels in TEST_LEVELS.items():
            for s, level in enumerate(levels):
                seed_ = seed + label * 3 + s
                rows.append(row(i, make_params(label, np.random.default_rng(seed_), level), SEVERITIES[s], seed_))
    return rows


if __name__ == "__main__":
    MANIFESTS.mkdir(exist_ok=True)
    for name, build in (("val", build_val), ("test", build_test)):
        rows = build()
        (MANIFESTS / f"{name}.json").write_text(json.dumps(rows))
        print(f"{name}: {len(rows)} entries")
