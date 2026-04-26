import argparse
import yaml
from importlib import import_module
from pathlib import Path
import datetime
# Added import
from gray_scott.utils import get_logger

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize logger
logger = get_logger()

def load_config(path):
    with open(path, "r") as f:
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
    parser.add_argument("--unified", action="store_false", help="Use unified training loop (if implemented)")
    args = parser.parse_args()

    config = load_config(args.config)
    config = deep_update(config, parse_set(args))

    problem = config["problem"]

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

    if args.unified:
        logger.info("Using unified training loop.")
        module = import_module("unified.train")
        module.main(config, run_dir)
    elif problem == "bratu":
        module = import_module("1d_bratu.train")
        module.main(config, run_dir)
    elif problem == "gray_scott":
        module = import_module("gray_scott.train")
        module.main(config, run_dir)
    else:
        logger.error(f"Unknown problem: {problem}")
        raise ValueError(f"Unknown problem: {problem}")

if __name__ == "__main__":
    main()