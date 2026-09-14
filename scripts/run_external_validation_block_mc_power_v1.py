import json
import random
from pathlib import Path
import itertools

CONFIG = Path("configs/external_validation_confirmatory_joint_power_simulation_v1.json")
OUT = Path("results/external_validation_block_mc_power_v1.json")

def simulate_cell(blocks, samples_per_block, reps=2000):
    joint_pass = 0
    ni_pass = 0
    abs_pass = 0

    # Block-level simulation placeholder:
    # each block is an independent provenance unit.
    for _ in range(reps):
        rng = random.Random(blocks * 1000 + samples_per_block * 10 + _)

        # baseline harmful recall degradation model
        fnr_reference = 0.05
        fnr_shift = rng.gauss(0.015, 0.04 / (blocks ** 0.5))

        fnr_test = fnr_reference + fnr_shift

        ni_ok = (fnr_test - fnr_reference) < 0.03
        abs_ok = fnr_test <= 0.10

        if ni_ok:
            ni_pass += 1
        if abs_ok:
            abs_pass += 1
        if ni_ok and abs_ok:
            joint_pass += 1

    return {
        "blocks": blocks,
        "samples_per_block": samples_per_block,
        "NI_power": round(ni_pass / reps, 4),
        "absolute_ceiling_power": round(abs_pass / reps, 4),
        "joint_power": round(joint_pass / reps, 4)
    }


def main():
    cfg = json.loads(CONFIG.read_text())

    results = []

    for blocks, samples in itertools.product(
        cfg["simulation_grid"]["provenance_blocks"],
        cfg["simulation_grid"]["samples_per_block"]
    ):
        results.append(simulate_cell(blocks, samples))

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))

    print("WROTE", OUT)
    print()
    print("=== BLOCK MC POWER SUMMARY ===")

    for r in results:
        if r["joint_power"] >= 0.80:
            print(r)


if __name__ == "__main__":
    main()
