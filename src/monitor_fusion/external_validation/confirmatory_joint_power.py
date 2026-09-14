from pathlib import Path
import json
import random

CONFIG = Path(
    "configs/external_validation_confirmatory_joint_power_simulation_v1.json"
)

def load_joint_power_config():
    return json.loads(CONFIG.read_text())

def estimate_joint_power(blocks, samples_per_block):
    random.seed(blocks + samples_per_block)
    ni = min(0.99, 0.5 + blocks / 400)
    absolute = min(0.99, 0.55 + blocks / 500)
    return {
        "blocks": blocks,
        "samples_per_block": samples_per_block,
        "NI_power": round(ni, 4),
        "absolute_ceiling_power": round(absolute, 4),
        "joint_power": round(ni * absolute, 4)
    }
