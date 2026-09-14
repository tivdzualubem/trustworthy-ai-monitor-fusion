import json
from pathlib import Path

INPUT = Path("results/external_validation_realistic_power_grid_v2.json")
OUTPUT = Path("results/external_validation_power_bottleneck_diagnosis_v1.json")


def main():
    rows = json.loads(INPUT.read_text())

    ni_fail = 0
    abs_fail = 0
    joint_fail = 0
    ni_pass = 0
    abs_pass = 0
    joint_pass = 0

    best = []

    for r in rows:
        ni = r["NI_power"] >= 0.80
        absolute = r["absolute_power"] >= 0.80
        joint = r["joint_power"] >= 0.80

        ni_pass += ni
        abs_pass += absolute
        joint_pass += joint

        ni_fail += not ni
        abs_fail += not absolute
        joint_fail += not joint

        if r["joint_power"] > 0.5:
            best.append(r)

    result = {
        "total_designs": len(rows),
        "criterion_counts": {
            "NI_pass": ni_pass,
            "absolute_pass": abs_pass,
            "joint_pass": joint_pass
        },
        "failure_counts": {
            "NI_failure": ni_fail,
            "absolute_failure": abs_fail,
            "joint_failure": joint_fail
        },
        "highest_joint_power_designs": sorted(
            best,
            key=lambda x: x["joint_power"],
            reverse=True
        )[:10],
        "interpretation_gate": {
            "next_question":
            "determine whether failure is driven by NI, absolute ceiling, or dependence assumptions"
        }
    }

    OUTPUT.write_text(json.dumps(result, indent=2))

    print("WROTE", OUTPUT)
    print("=== BOTTLENECK SUMMARY ===")
    print(result["criterion_counts"])
    print(result["failure_counts"])
    print("TOP DESIGNS")
    for x in result["highest_joint_power_designs"][:5]:
        print(x)


if __name__ == "__main__":
    main()
