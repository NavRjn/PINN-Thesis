import argparse
import yaml
import json
import importlib
from pathlib import Path
from gray_scott.utils import get_logger
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

logger = get_logger()


def get_latest_run(problem):
    base = Path("outputs") / problem
    if not base.exists():
        logger.error(f"Problem directory not found: {base}")
        raise ValueError(f"No outputs for {problem}")

    runs = sorted(base.glob("*"))
    if not runs:
        logger.error(f"No runs found for {problem}.")
        raise ValueError(f"No runs found for {problem}.")
    return runs[-1]


def load_config(run_dir: Path):
    """Loads the config saved during the training run."""
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    # Fallback to legacy params.json if necessary
    legacy_path = run_dir / "params.json"
    if legacy_path.exists():
        with open(legacy_path, "r") as f:
            return json.load(f)

    return {}


def main():
    parser = argparse.ArgumentParser(description="Unified Visualization CLI")
    parser.add_argument("--problem", type=str, default="gray_scott", help="Problem name (folder name)")
    parser.add_argument("--run", type=str, help="Specific timestamp folder")
    parser.add_argument("--last", action="store_true", help="Use the most recent run")

    # Visualization Hyperparameters
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--z_end", type=float, default=10.0)
    parser.add_argument("--z_start", type=float, default=None)
    parser.add_argument("--ckpt_type", type=str, default="final")
    parser.add_argument("--n", type=int, default=100)

    args = parser.parse_args()

    # 1. Resolve Directory
    try:
        if args.last:
            run_dir = get_latest_run(args.problem)
        elif args.run:
            run_dir = Path("outputs") / args.problem / args.run
        else:
            logger.error("Must provide --last or --run")
            return

        if not run_dir.exists():
            logger.error(f"Run directory not found: {run_dir}")
            return
    except ValueError as e:
        logger.error(str(e))
        return

    logger.info(f"Visualizing results from: {run_dir}")

    # 2. Prepare Config for API
    # Combine CLI overrides with the saved run config
    run_config = load_config(run_dir)
    viz_overrides = {
        "num_steps": args.num_steps,
        "z_start_val": args.z_start if args.z_start is not None else -args.z_end,
        "z_end_val": args.z_end,
        "n": args.n,
        "ckpt_type": args.ckpt_type
    }
    # Merge: CLI overrides take precedence for the visualization session
    full_config = {**run_config, **viz_overrides}

    try:
        # Try to use the new API hook
        api = importlib.import_module(f"{args.problem}.api").API # import as class, without instantiating
        logger.debug(f"Import api from {args.problem}.api successful")
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Using unified API for {args.problem}")
        api.post_process_visualize(run_dir, full_config, device)

    except (ModuleNotFoundError) as e:

        logger.error(f"API visualization failed for {args.problem}: {e}")
        # 4. Fallback to Legacy Scripts
        logger.warning(f"No api.post_process_visualize found for {args.problem}. Falling back to legacy scripts.")

        if args.problem == "bratu":
            legacy = importlib.import_module("1d_bratu.visualize")
            legacy.main({}, run_dir)
        elif args.problem == "gray_scott":
            legacy = importlib.import_module("gray_scott.visualize")
            legacy.main(viz_overrides, run_dir)
        else:
            logger.error(f"Cannot visualize {args.problem}: No API or legacy script found.")

    print("Visualization complete.")


if __name__ == "__main__":
    main()