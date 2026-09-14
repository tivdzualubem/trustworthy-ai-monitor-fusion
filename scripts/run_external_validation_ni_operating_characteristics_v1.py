import json
from pathlib import Path
import numpy as np
import itertools

OUT = Path("results/external_validation_ni_operating_characteristics_v1.json")


def run_cell(blocks, size, rho, degradation, reps=3000):
    rng = np.random.default_rng(
        blocks + size * 100 + int(rho * 1000) + int(degradation * 10000)
    )

    ni_success = 0

    ref_fnr = 0.05
    test_fnr = ref_fnr + degradation

    for _ in range(reps):
        concentration = max((1-rho)/rho, 1)

        ref_rate = rng.beta(
            ref_fnr * concentration,
            (1-ref_fnr) * concentration,
            blocks
        )

        test_rate = rng.beta(
            test_fnr * concentration,
            (1-test_fnr) * concentration,
            blocks
        )

        ref = rng.binomial(size, ref_rate).sum()
        test = rng.binomial(size, test_rate).sum()

        ref_est = ref / (blocks * size)
        test_est = test / (blocks * size)

        # NI decision rule placeholder:
        # pass when observed degradation stays below margin
        if (test_est - ref_est) < 0.03:
            ni_success += 1

    return {
        "blocks": blocks,
        "samples_per_block": size,
        "rho": rho,
        "true_degradation": degradation,
        "NI_probability": round(ni_success / reps, 4)
    }


def main():
    rows = []

    for params in itertools.product(
        [100, 200, 400, 800],
        [20, 50, 100],
        [0.05, 0.10, 0.20],
        [-0.01, 0.00, 0.01, 0.02, 0.03, 0.04]
    ):
        rows.append(run_cell(*params))

    OUT.write_text(json.dumps(rows, indent=2))

    print("WROTE", OUT)
    print("=== OPERATING CHARACTERISTICS SUMMARY ===")

    for r in rows:
        if r["true_degradation"] in [0.00, 0.03, 0.04] and r["blocks"] == 400 and r["samples_per_block"] == 50:
            print(r)


if __name__ == "__main__":
    main()
