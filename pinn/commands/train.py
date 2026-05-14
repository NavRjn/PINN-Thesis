import typer
import yaml

app = typer.Typer()

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

def parse_set(overrides_set, logger=None):

    overrides = {}
    if overrides_set:
        for item in overrides_set:
            key, value = item.split("=")
            try:
                value = yaml.safe_load(value)
            except:
                if logger: logger.warn(f"Invalid override: {key}:{value}; using string.")
            keys = key.split(".")
            d = overrides
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            d[keys[-1]] = value
    return overrides

def train_from_config(config_name, overrides=None):
    from importlib import import_module
    from pathlib import Path
    import datetime
    from core.utils import get_logger
    logger = get_logger()

    if overrides is None:
        overrides = dict()
    config = load_config(config_name)
    config = deep_update(config, overrides)

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

@app.callback(invoke_without_command=True)
def train(config = typer.Argument(...,
                                  help="Path to config file relative to configs/"),
          override_set = typer.Option(None,"--set",
                             "-s", help="Override config values, e.g. --set training.lr=0.001")
          ):

    train_from_config(config, parse_set(override_set))


if __name__ == "__main__":
    app()