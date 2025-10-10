import json
from pathlib import Path
import joblib
from src.config import settings

MODEL_DIR = Path(settings.model_dir)
MODEL_DIR.mkdir(exist_ok=True, parents=True)
REGISTRY_FILE = MODEL_DIR / "registry.json"

def get_registry() -> dict:
    if not REGISTRY_FILE.exists():
        return {}
    try:
        return json.loads(REGISTRY_FILE.read_text())
    except Exception:
        return {}

def register_registry(mapping: dict):
    REGISTRY_FILE.write_text(json.dumps(mapping, indent=2))

def load_model(name: str):
    reg = get_registry()
    if name not in reg:
        raise KeyError(f"model {name} not in registry")
    path = MODEL_DIR / reg[name]
    if not path.exists():
        raise FileNotFoundError(f"model file not found: {path}")
    return joblib.load(path)
