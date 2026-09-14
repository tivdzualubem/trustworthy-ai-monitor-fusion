import json
from pathlib import Path
import numpy as np
import itertools

CFG = Path("configs/external_validation_realistic_power_grid_v2.json")
OUT = Path("results/external_validation_realistic_power_grid_v2.json")


def beta_binomial_errors(rng, blocks, size, mean, rho):
    concentration = max((1-rho)/rho, 1)
    a = max(mean * concentration, 0.1)
    b = max((1-mean) * concentration, 0.1)

    rates = rng.beta(a, b, blocks)
    return rng.binomial(size, rates).sum()


def evaluate(blocks, size, rho, ref_fnr, reps=2000):
    rng = np.random.default_rng(
        blocks + size*100 + int(rho*1000) + int(ref_fnr*10000)
    )

    ni = 0
    absolute = 0
    joint = 0

    test_fnr = ref_fnr + 0.03

    for _ in range(reps):
        ref = beta_binomial_errors(
            rng, blocks, size, ref_fnr, rho
        )
        test = beta_binomial_errors(
            rng, blocks, size, test_fnr, rho
        )

        total = blocks * size

        ref_rate = ref / total
        test_rate = test / total

        ni_ok = (test_rate - ref_rate) < 0.03
        abs_ok = test_rate <= 0.10

        ni += ni_ok
        absolute += abs_ok
        joint += ni_ok and abs_ok

    return {
        "blocks": blocks,
        "samples_per_block": size,
        "rho": rho,
        "reference_FNR": ref_fnr,
        "NI_power": round(ni/reps,4),
        "absolute_power": round(absolute/reps,4),
        "joint_power": round(joint/reps,4)
    }


def main():
    cfg = json.loads(CFG.read_text())
    rows=[]

    for x in itertools.product(
        cfg["grid"]["blocks"],
        cfg["grid"]["samples_per_block"],
        cfg["grid"]["rho"],
        cfg["scenarios"]["relative_NI_boundary"]["reference_FNR"]
    ):
        rows.append(evaluate(*x))

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2))

    print("WROTE", OUT)
    print("=== DESIGNS REACHING TARGET ===")

    found=False
    for r in rows:
        if r["joint_power"] >= cfg["target_joint_power"]:
            print(r)
            found=True

    if not found:
        print("NO DESIGN REACHED TARGET POWER")


if __name__ == "__main__":
    main()
