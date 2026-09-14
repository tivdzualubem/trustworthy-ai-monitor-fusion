import json
from pathlib import Path
import itertools

CONFIG = Path("configs/external_validation_confirmatory_joint_power_simulation_v1.json")
OUT = Path("results/external_validation_confirmatory_joint_power_v1.json")

def simulate_power(blocks, samples):
    # Placeholder planning model. Replace with validated MC engine after design freeze.
    ni = min(0.99, 0.55 + blocks / 350)
    absolute = min(0.99, 0.60 + blocks / 400)
    return {
        "blocks": blocks,
        "samples_per_block": samples,
        "ni_power": round(ni, 4),
        "absolute_ceiling_power": round(absolute, 4),
        "joint_power": round(ni * absolute, 4)
    }

def main():
    cfg = json.loads(CONFIG.read_text())
    rows = []

    for blocks, samples in itertools.product(
        cfg["simulation_grid"]["provenance_blocks"],
        cfg["simulation_grid"]["samples_per_block"]
    ):
        rows.append(simulate_power(blocks, samples))

    recommended = [
        r for r in rows
        if r["joint_power"] >= 0.80
    ]

    result = {
        "grid_results": rows,
        "recommended_designs": recommended[:5]
    }

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))

    print("WROTE", OUT)
    print("=== JOINT POWER SUMMARY ===")
    for r in recommended[:5]:
        print(r)

if __name__ == "__main__":
    main()
