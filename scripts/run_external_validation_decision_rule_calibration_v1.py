import json
from pathlib import Path
import numpy as np
import itertools

OUT = Path("results/external_validation_decision_rule_calibration_v1.json")

def simulate(blocks, size, rho, degradation, reps=3000):
    rng = np.random.default_rng(blocks + size*100 + int(rho*1000))
    accepted = 0

    ref_fnr = 0.05
    test_fnr = ref_fnr + degradation
    concentration = max((1-rho)/rho, 1)

    for _ in range(reps):
        ref_rates = rng.beta(ref_fnr*concentration,
                             (1-ref_fnr)*concentration,
                             blocks)
        test_rates = rng.beta(test_fnr*concentration,
                              (1-test_fnr)*concentration,
                              blocks)

        ref = rng.binomial(size, ref_rates).sum()
        test = rng.binomial(size, test_rates).sum()

        total = blocks * size
        ref_est = ref / total
        test_est = test / total

        if (test_est-ref_est) < 0.03 and test_est <= 0.10:
            accepted += 1

    return {
        "blocks": blocks,
        "samples_per_block": size,
        "rho": rho,
        "true_degradation": degradation,
        "acceptance_probability": round(accepted/reps, 4)
    }

def main():
    rows = []
    for params in itertools.product(
        [200, 400, 800],
        [50, 100],
        [0.05, 0.10, 0.20],
        [0.00, 0.01, 0.02, 0.03, 0.04]
    ):
        rows.append(simulate(*params))

    OUT.write_text(json.dumps(rows, indent=2))
    print("WROTE", OUT)
    print("=== DECISION RULE SUMMARY ===")

    for r in rows:
        if r["blocks"] == 400 and r["samples_per_block"] == 100:
            print(r)

if __name__ == "__main__":
    main()
