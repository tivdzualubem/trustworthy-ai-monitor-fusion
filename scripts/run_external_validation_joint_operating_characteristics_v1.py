import json
from pathlib import Path
import itertools
import numpy as np

OUT = Path("results/external_validation_joint_operating_characteristics_v1.json")


def simulate(blocks, size, rho, degradation, reps=3000):
    rng = np.random.default_rng(
        blocks + size * 100 + int(rho * 1000) + int(degradation * 10000)
    )

    ni = 0
    absolute = 0
    joint = 0

    ref_fnr = 0.05
    test_fnr = ref_fnr + degradation

    concentration = max((1-rho)/rho, 1)

    for _ in range(reps):
        ref_rates = rng.beta(
            ref_fnr * concentration,
            (1-ref_fnr) * concentration,
            blocks
        )

        test_rates = rng.beta(
            test_fnr * concentration,
            (1-test_fnr) * concentration,
            blocks
        )

        ref_errors = rng.binomial(size, ref_rates).sum()
        test_errors = rng.binomial(size, test_rates).sum()

        total = blocks * size

        ref_est = ref_errors / total
        test_est = test_errors / total

        ni_ok = (test_est - ref_est) < 0.03
        abs_ok = test_est <= 0.10

        ni += ni_ok
        absolute += abs_ok
        joint += ni_ok and abs_ok

    return {
        "blocks": blocks,
        "samples_per_block": size,
        "rho": rho,
        "degradation": degradation,
        "NI_power": round(ni/reps, 4),
        "absolute_power": round(absolute/reps, 4),
        "joint_power": round(joint/reps, 4)
    }


def main():
    rows = []

    for params in itertools.product(
        [100, 200, 400, 800],
        [20, 50, 100],
        [0.05, 0.10, 0.20],
        [0.00, 0.01, 0.02, 0.03, 0.04]
    ):
        rows.append(simulate(*params))

    OUT.write_text(json.dumps(rows, indent=2))

    print("WROTE", OUT)
    print("=== JOINT SUMMARY ===")

    for r in rows:
        if r["blocks"] == 400 and r["samples_per_block"] == 50:
            print(r)


if __name__ == "__main__":
    main()
