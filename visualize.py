import argparse
from pathlib import Path
from importlib import import_module
# Added import
from gray_scott.utils import get_logger

# Initialize logger
logger = get_logger()

def get_latest_run(problem):
    base = Path("outputs") / problem
    runs = sorted(base.glob("*"))
    if not runs:
        logger.error(f"No runs found for {problem}.")
        raise ValueError(f"No runs found for {problem}.")
    return runs[-1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="gray_scott")
    parser.add_argument("--run", type=str)
    parser.add_argument("--last", action="store_true")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of steps for latent space analysis")
    parser.add_argument("--z_end", type=float, default=10.0, help="Limit value of latent z for analysis")
    parser.add_argument("--z_start", type=float, default=None, help="Optional custom starting value")
    parser.add_argument("--ckpt_type", type=str, default="final", help="Type of checkpoint to load")
    args = parser.parse_args()

    if args.last:
        run_dir = get_latest_run(args.problem)
    elif args.run:
        run_dir = Path("outputs") / args.problem / args.run
        if not run_dir.exists():
            logger.error(f"Specified run directory does not exist: {run_dir}")
            raise ValueError(f"Run directory not found: {run_dir}")
    else:
        logger.error("User did not provide --last or --run")
        raise ValueError("Use --last or --run")

    if args.ckpt_type not in ["final", "best"]:
        logger.error(f"Invalid checkpoint type: {args.ckpt_type}")
        raise ValueError("ckpt_type must be 'final' or 'best'")

    # Replaced print with logger.info
    logger.info(f"Visualizing results from: {run_dir}")

    config = {
        "num_steps": args.num_steps,
        "z_start_val": args.z_start if args.z_start is not None else -args.z_end,
        "z_end_val": args.z_end,
        "ckpt_type": args.ckpt_type
    }

    logger.debug(f"Visualization config: {config}")

    if args.problem == "bratu":
        module = import_module("1d_bratu.visualize")
        module.main({}, run_dir)
    elif args.problem == "gray_scott":
        module = import_module("gray_scott.visualize")
        module.main(config, run_dir)
    else:
        logger.error(f"Unknown problem: {args.problem}")
        raise ValueError("Unknown problem")

if __name__ == "__main__":
    main()