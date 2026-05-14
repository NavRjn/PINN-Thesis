from pathlib import Path
import typer




app = typer.Typer()

# remove later
EXCLUDED_DIRS = {
    "core",
    "pinn",
    "configs",
    "outputs",
    "templates",
    "__pycache__",
    ".git",
}


def discover_problems():
    """
    Dynamically discover valid problem modules.
    """
    root = Path(".")
    problems = []
    for path in root.iterdir():
        if not path.is_dir():
            continue

        if path.name in EXCLUDED_DIRS:
            continue

        if (path / "api.py").exists():
            problems.append(path.name)

    return sorted(problems)


# =========================================================
# RUN DISCOVERY
# =========================================================

def get_latest_run(problem: str):
    """
    Get latest run for a specific problem.
    """

    base = Path("outputs") / problem
    if not base.exists():
        raise ValueError(f"No outputs found for problem: {problem}")

    runs = sorted(base.glob("*"))
    if not runs:
        raise ValueError(f"No runs found for problem: {problem}")

    return runs[-1]


def get_latest_run_global():
    """
    Search ALL problems for latest run globally.
    """
    latest_run = None
    latest_time = None

    for problem in discover_problems():

        base = Path("outputs") / problem
        if not base.exists():
            continue

        runs = sorted(base.glob("*"))
        if not runs:
            continue

        newest = runs[-1]
        timestamp = newest.name
        if latest_time is None or timestamp > latest_time:
            latest_time = timestamp
            latest_run = newest

    if latest_run is None:
        raise ValueError("No runs found across any problem")

    return latest_run


# =========================================================
# CONFIG LOADING
# =========================================================

def load_config(run_dir: Path):
    import json
    import yaml

    config_path = run_dir / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    legacy_path = run_dir / "params.json"
    if legacy_path.exists():
        with open(legacy_path, "r") as f:
            return json.load(f)

    return {}


# =========================================================
# CORE VISUALIZATION LOGIC
# =========================================================

def visualize_run(
    problem: str,
    run_dir: Path,
    viz_overrides: dict,
):
    """
    Reusable visualization entrypoint.
    """
    import importlib
    import torch
    from core.utils import get_logger

    logger = get_logger()

    logger.info(f"Visualizing results from: {run_dir}")
    run_config = load_config(run_dir)
    full_config = {
        **run_config,
        **viz_overrides,
    }

    try:
        api = importlib.import_module( f"{problem}.api" ).API
        logger.debug(f"Imported API from {problem}.api")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Using unified API for {problem}")
        api.post_process_visualize(run_dir, full_config, device)

    except ModuleNotFoundError as e:
        logger.error(f"API visualization failed for {problem}: {e}")
        logger.warning("Falling back to legacy visualization")


    logger.info("Visualization complete")


# =========================================================
# CLI
# =========================================================

@app.callback(invoke_without_command=True)
def viz(
    problem: str = typer.Argument(None, help="Problem name"),
    run: str = typer.Option(None, "--run", help="Specific run ID",),
    last: bool = typer.Option(False,"--last", help="Use latest run",),
    num_steps: int = typer.Option(50),
    z_end: float = typer.Option(10.0),
    z_start: float = typer.Option(None),
    ckpt_type: str = typer.Option("final"),
    n: int = typer.Option(100),
):
    """
    Visualize PINN outputs.
    """
    from core.utils import get_logger

    logger = get_logger()

    try:
        # Resolve run directory
        if last:
            if problem is None:
                run_dir = get_latest_run_global()
                problem = run_dir.parent.name
                logger.info(f"Auto-detected latest problem: {problem}")
            else:
                run_dir = get_latest_run(problem)
        elif run:
            if problem is None:
                raise ValueError("Problem must be provided with --run")
            run_dir = Path("outputs") / problem/ run
        else:
            raise ValueError("Provide either --last or --run")

        # Validation
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        # Visualization config
        viz_overrides = {
            "num_steps": num_steps,
            "z_start_val": z_start if z_start is not None else -z_end,
            "z_end_val": z_end,
            "n": n,
            "ckpt_type": ckpt_type,
        }

        visualize_run( problem, run_dir, viz_overrides,)

    except Exception as e:
        logger.error(str(e))
        raise typer.Exit(code=1)


# =========================================================
# DIRECT EXECUTION
# =========================================================
if __name__ == "__main__":
    app()