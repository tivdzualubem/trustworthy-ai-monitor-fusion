from pathlib import Path
import json

CONFIG = Path("configs/external_validation_confirmatory_power_design_v1.json")

def load_power_design():
    return json.loads(CONFIG.read_text())
