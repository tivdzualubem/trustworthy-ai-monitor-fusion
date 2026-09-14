import json
from pathlib import Path
import numpy as np
import itertools

CONFIG = Path("configs/external_validation_confirmatory_joint_power_simulation_v1.json")
OUT = Path("results/external_validation_realistic_block_mc_power_v1.json")

def beta_binomial_blocks(rng, n_blocks, block_size, mean, rho):
    # Block random effects approximation:
    # beta-binomial style correlated Bernoulli outcomes
    a = mean * (1-rho) / rho
    b = (1-mean) * (1-rho) / rho
    rates = rng.beta(a, b, size=n_blocks)
    return rng.binomial(block_size, rates).sum()

def run_cell(blocks, block_size, reps=5000):
    rng = np.random.default_rng(blocks * 100 + block_size)

    ni_success = 0
    abs_success = 0
    joint_success = 0

    # Hard confirmatory scenarios:
    # reference FNR = 0.07
    # shifted FNR = 0.10 (exact 3pp NI boundary)
    ref_fnr = 0.07
    test_fnr = 0.10
    rho = 0.10

    for _ in range(reps):
        ref_errors = beta_binomial_blocks(
            rng, blocks, block_size, ref_fnr, rho
        )
        test_errors = beta_binomial_blocks(
            rng, blocks, block_size, test_fnr, rho
        )

        total = blocks * block_size
        ref_rate = ref_errors / total
        test_rate = test_errors / total

        ni = (test_rate - ref_rate) < 0.03
        absolute = test_rate <= 0.10

        ni_success += ni
        abs_success += absolute
        joint_success += (ni and absolute)

    return {
        "blocks": blocks,
        "block_size": block_size,
        "NI_power": round(ni_success/reps, 4),
        "absolute_ceiling_power": round(abs_success/reps, 4),
        "joint_power": round(joint_success/reps, 4)
    }

def main():
    cfg = json.loads(CONFIG.read_text())
    results = []

    for blocks, size in itertools.product(
        cfg["simulation_grid"]["provenance_blocks"],
        cfg["simulation_grid"]["samples_per_block"]
    ):
        results.append(run_cell(blocks, size))

    OUT.write_text(json.dumps(results, indent=2))

    print("WROTE", OUT)
    print("\n=== REALISTIC BLOCK MC SUMMARY ===")
    for row in results:
        if row["joint_power"] >= 0.80:
            print(row)

if __name__ == "__main__":
    main()
