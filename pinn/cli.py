import typer
from pathlib import Path
from rich.console import Console
import os
import sys

# Temporary hack to ensure CLI works when run from the root of the repo
REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path.cwd()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from pinn.registry import (
    list_problems,
    list_runs,
    load_config,
    load_losses,
)
# from pinn.commands.train import app as train_app
# from pinn.commands.visualize import app as viz_app
# from pinn.commands.add import app as add_app
# from pinn.commands.init import app as init_app

from pinn.commands.train import train as train_cmd
from pinn.commands.visualize import viz as viz_cmd
from pinn.commands.add import add as add_cmd
from pinn.commands.init import init as init_cmd

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = typer.Typer()
console = Console()

# Stay away from subapps for now.
# app.add_typer(train_app, name="train")
# app.add_typer(viz_app, name="viz")
# app.add_typer(add_app, name="add")
# app.add_typer(init_app, name="init")

# Prefer direct commands for simplicity in this early stage.
app.command()(train_cmd)
app.command()(viz_cmd)
app.command()(add_cmd)
app.command()(init_cmd)


@app.command()
def list():
    """
    List all available problems with at least one run.
    """
    problems = list_problems()
    if not problems:
        console.print("No problems found.")
        return

    console.print("\n[bold]Available problems:[/bold]\n")
    for p in problems:
        runs = list_runs(p)
        console.print(f"- {p} ({len(runs)} runs)")

@app.command()
def runs(problem: str):
    """
    List runs for a given problem.
    """
    run_dirs = list_runs(problem)

    if not run_dirs:
        console.print(f"No runs found for {problem}")
        return

    console.print(f"\n[bold]Runs for {problem}:[/bold]\n")

    for r in run_dirs:
        console.print(f"- {r.name}")

@app.command()
def info(problem: str, run: str = None):
    """
    Show metadata summary for a run (or latest if not specified).
    """

    run_dirs = list_runs(problem)

    if not run_dirs:
        console.print(f"No runs found for {problem}")
        return

    if run is None:
        run_dir = run_dirs[-1]
    else:
        run_dir = Path("outputs") / problem / run

    if not run_dir.exists():
        console.print(f"Run not found: {run_dir}")
        return

    config = load_config(run_dir)
    losses = load_losses(run_dir)

    # -------------------------
    # Extract key metadata
    # -------------------------

    model = config.get("model", {})
    training = config.get("training", {})

    last_loss = None
    if isinstance(losses, dict):
        try:
            obj = losses.get("obj", {})
            if isinstance(obj, dict) and len(obj) > 0:
                last_loss = list(obj.values())[-1]
        except Exception:
            pass

    # -------------------------
    # Display
    # -------------------------

    console.print("\n[bold cyan]Run Info[/bold cyan]\n")

    console.print(f"[bold]Run:[/bold] {run_dir.name}")
    console.print(f"[bold]Problem:[/bold] {config.get('problem', problem)}")

    console.print("\n[bold]Model[/bold]")
    console.print(f"  name: {model.get('name')}")
    console.print(f"  arch: {model.get('arch')}")
    console.print(f"  nx: {model.get('nx')}, ny: {model.get('ny')}, nz: {model.get('nz')}")

    console.print("\n[bold]Training[/bold]")
    console.print(f"  method: {training.get('method')}")
    console.print(f"  lr: {training.get('lr')}")
    console.print(f"  n_iters: {training.get('n')}")
    console.print(f"  batch: {training.get('bz')}")

    console.print("\n[bold]Physics[/bold]")
    physics = config.get("physics", {})
    for k, v in physics.items():
        if k != "bounds":
            console.print(f"  {k}: {v}")

    if last_loss is not None:
        console.print(f"\n[bold green]Last loss:[/bold green] {last_loss}")

if __name__ == "__main__":
    app()