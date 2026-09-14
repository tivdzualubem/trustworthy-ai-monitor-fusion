from pathlib import Path
import json

CONFIG = Path("configs/external_validation_confirmatory_design_v3.json")

def load_confirmatory_design():
    return json.loads(CONFIG.read_text())
