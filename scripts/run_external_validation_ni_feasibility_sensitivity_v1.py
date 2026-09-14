import json
from pathlib import Path
import itertools
import numpy as np

OUT = Path("results/external_validation_ni_feasibility_sensitivity_v1.json")


def simulate(blocks, size, rho, degradation, reps=3000):
    rng = np.random.default_rng(
        blocks + size * 100 + int(rho * 1000) + int(degradation * 10000)
    )

    success = 0

    ref_fnr = 0.05
    test_fnr = ref_fnr + degradation

    for _ in range(reps):
        concentration = max((1-rho)/rho, 1)

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

        ref_est = ref_errors / (blocks * size)
        test_est = test_errors / (blocks * size)

        if (test_est - ref_est) < 0.03:
            success += 1

    return {
        "blocks": blocks,
        "samples_per_block": size,
        "rho": rho,
        "true_FNR_degradation": degradation,
        "NI_power": round(success / reps, 4)
    }


def main():
    rows = []

    for params in itertools.product(
        [100, 200, 400, 800],
        [20, 50, 100],
        [0.05, 0.10, 0.20],
        [0.00, 0.01, 0.02, 0.03]
    ):
        rows.append(simulate(*params))

    OUT.write_text(json.dumps(rows, indent=2))

    print("WROTE", OUT)
    print("=== BEST NI DESIGNS ===")

    for r in sorted(
        rows,
        key=lambda x: x["NI_power"],
        reverse=True
    )[:10]:
        print(r)


if __name__ == "__main__":
    main()
