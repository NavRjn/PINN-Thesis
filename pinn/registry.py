from pathlib import Path
import yaml
import json

OUTPUTS = Path("outputs")


def list_problems():
    if not OUTPUTS.exists():
        return []
    return sorted([p.name for p in OUTPUTS.iterdir() if p.is_dir()])


def list_runs(problem: str):
    base = OUTPUTS / problem
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])


def load_config(run_dir: Path):
    path = run_dir / "config.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text())


def load_losses(run_dir: Path):
    path = run_dir / "losses.json"
    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text())
    except Exception:
        return {}