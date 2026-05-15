import typer
from pathlib import Path
import importlib.resources as pkg_resources

app = typer.Typer()

REPO_ROOT = Path(__file__).resolve().parents[1]
CWD = Path.cwd()


# @app.callback(invoke_without_command=True)
def init(
        name: str = typer.Argument(..., help="Name of the new research project directory")
):
    """
    Initialize a new PINN research workspace.
    """
    project_dir = CWD / name
    print("🔧 Choosing project location: ", project_dir)
    if project_dir.exists():
        typer.echo(f"[ERROR] Directory '{name}' already exists.", err=True)
        raise typer.Exit(code=1)

    # 1. Create directory structure
    project_dir.mkdir()
    (project_dir / "configs").mkdir()
    (project_dir / "outputs").mkdir()

    # 2. Extract base configuration from packaged templates
    try:
        # Python 3.9+ way to read packaged data
        import pinn.templates.zero as templates_module
        template_files = pkg_resources.files(templates_module)

        # We assume you added a generic.yaml inside pinn/templates/zero/
        base_cfg = template_files.joinpath("generic.yaml").read_text()
        (project_dir / "configs" / "generic.yaml").write_text(base_cfg)

    except Exception as e:
        typer.echo(f"[WARN] Could not copy default config: {e}")

    # 3. Create .env file for WandB
    typer.echo("🚀 Workspace created!")
    wandb_key = typer.prompt("Enter Weights & Biases API Key (press Enter to skip)", default="")

    if wandb_key:
        (project_dir / ".env").write_text(f"WANDB_API_KEY={wandb_key}\n")
        typer.echo("✅ Saved W&B credentials.")

    typer.echo(f"\nNext steps:\n  cd {name}\n  pinn add my_first_pde")