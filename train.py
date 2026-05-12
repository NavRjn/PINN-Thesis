import argparse
import yaml
from importlib import import_module
from pathlib import Path
import datetime
# Added import
from core.utils import get_logger

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize logger
logger = get_logger()

def load_config(path):
    with open("configs/"+path, "r") as f:
        return yaml.safe_load(f)

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def parse_set(args):
    overrides = {}
    if args.set:
        for item in args.set:
            key, value = item.split("=")
            try:
                value = eval(value)
            except:
                pass
            keys = key.split(".")
            d = overrides
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
    return overrides

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--set", nargs="+")
    args = parser.parse_args()

    config = load_config(args.config)
    config = deep_update(config, parse_set(args))

    problem = config["problem"]
    problem_path = Path(problem)
    if not problem_path.exists():
        logger.error(f"Problem {problem} does not exist")
        print(f"Invalid problem: {problem}; not found.")
        return
    elif not (problem_path / "api.py").exists():
        logger.error(f"Invalid {problem} does not contain. api.py")
        print(f"Invalid problem: {problem}; implement api.py first.")
        return

        # Create run directory
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("outputs") / problem / run_id

    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Replaced print with logger.info
    logger.info(f"Training run initiated: {run_dir}")
    logger.info("Using unified training loop.")

    module = import_module("core.train")
    module.main(config, run_dir, logger)

if __name__ == "__main__":
    main()