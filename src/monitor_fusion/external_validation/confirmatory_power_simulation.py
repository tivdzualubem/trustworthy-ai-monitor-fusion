from pathlib import Path
import json

CONFIG = Path("configs/external_validation_confirmatory_power_simulation_v1.json")

def load_power_simulation():
    return json.loads(CONFIG.read_text())
