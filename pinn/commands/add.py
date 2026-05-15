from pathlib import Path

import typer

REPO_ROOT = Path(__file__).resolve().parents[2]
CWD = Path.cwd()

# PATHS
TEMPLATE_DIR = REPO_ROOT / "pinn" / "templates" / "zero"

# problem folders live at repository root
PROBLEMS_DIR = CWD
CONFIGS_DIR = CWD / "configs"

app = typer.Typer()

# UTILITIES
def render_template(path: Path, replacements: dict) -> str:
    txt = path.read_text()
    for k, v in replacements.items():
        txt = txt.replace(f"{{{{ {k} }}}}", v)

    return txt

def file_plan(name: str):
    """
    Compute what will be created/overwritten.
    """
    problem_dir = PROBLEMS_DIR / name
    config_path = CONFIGS_DIR / f"{name}.yaml"
    tpl_files = list(TEMPLATE_DIR.glob("*.tpl"))

    planned = []
    planned.append(("CONFIG", config_path, config_path.exists()))

    for tpl in tpl_files:
        if tpl.name == "README.tpl":
            out = problem_dir / "README.md"
        else:
            out = problem_dir / (tpl.stem + ".py")

        planned.append(("PY", out, out.exists()))

    planned.append(("DIR", problem_dir, problem_dir.exists()))

    return planned

def print_plan(plan):
    typer.echo("\n=== ACTION PLAN ===")

    for typ, path, exists in plan:
        status = ("OVERWRITE ⚠" if exists else "CREATE")
        typer.echo(f"{typ:6} | {status:12} | {path}")

    typer.echo("===================\n")

def git_add(files):
    import subprocess

    paths = [str(f) for f in files]
    subprocess.run(["git", "add", *paths], check=False)

def print_next_steps(
    problem_dir: Path,
    config_path: Path,
):

    api_path = problem_dir / "api.py"
    problem_path = problem_dir / "problem.py"
    readme_path = problem_dir / "README.md"
    typer.echo("\n" + "=" * 60)

    typer.echo("🚀 Problem scaffold created successfully!")
    typer.echo("=" * 60)

    typer.echo("\n📌 NEXT STEPS\n")
    typer.echo("1) Configure experiment:")
    typer.echo(f'   👉 "{config_path.resolve()}"')
    typer.echo("   - change model/training/physics params\n")

    typer.echo("2) Define problem logic:")
    typer.echo(f'   👉 "{problem_path.resolve()}"')
    typer.echo("   - implement grid_sampler()")
    typer.echo("   - implement loss_fn(model, batch)\n")

    typer.echo("3) Optional API customization:")
    typer.echo(f'   👉 "{api_path.resolve()}"')
    typer.echo("   - only if custom behavior is needed\n")

    typer.echo("4) Documentation:")
    typer.echo(f'   👉 "{readme_path.resolve()}"')

    typer.echo("\n💡 Tip:")
    typer.echo("   Start by editing ONLY problem.py first.\n")
    typer.echo("=" * 60 + "\n")

# CORE LOGIC
def create_problem(
    name: str,
    overwrite: bool = False,
    dry_run: bool = False,
    yes: bool = False,
    force_git: bool = False,
):
    """
    Reusable scaffold creation logic.
    """

    plan = file_plan(name)
    print_plan(plan)
    if dry_run:
        typer.echo("[DRY RUN] No files written.")
        return

    # Confirmation
    if not yes:
        confirmed = typer.confirm("Proceed with creation?", default=False,)
        if not confirmed:
            typer.echo("Aborted.")
            return

    created_files = []
    problem_dir = PROBLEMS_DIR / name
    problem_dir.mkdir(parents=True, exist_ok=True)

    # CONFIG
    base_cfg = CONFIGS_DIR / "generic.yaml"
    cfg_text = base_cfg.read_text()
    cfg_text = cfg_text.replace('problem: "generic_problem"',f'problem: "{name}"')

    cfg_path = CONFIGS_DIR / f"{name}.yaml"

    if cfg_path.exists() and not overwrite:
        typer.echo(f"[SKIP] {cfg_path}")
    else:
        cfg_path.write_text(cfg_text)
        created_files.append(cfg_path)
        typer.echo(f"[OK] {cfg_path}")

    # TEMPLATE FILES
    replacements = {"problem_name": name}

    for tpl in TEMPLATE_DIR.glob("*.tpl"):
        if tpl.name == "README.tpl":
            out_path = problem_dir / "README.md"
        else:
            out_path = problem_dir / (tpl.stem + ".py")

        if out_path.exists() and not overwrite:
            typer.echo(f"[SKIP] {out_path}")
            continue

        content = render_template(tpl, replacements)
        out_path.write_text(content)
        created_files.append(out_path)
        typer.echo(f"[OK] {out_path}")

    # __init__.py
    init_file = problem_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")
        created_files.append(init_file)

    # GIT ADD
    should_git_add = force_git
    if not force_git and not yes:
        should_git_add = typer.confirm("Add generated files to git?", default=True)

    if should_git_add:
        git_add(created_files)
        typer.echo("[OK] added to git")

    # NEXT STEPS
    print_next_steps(problem_dir, cfg_path)

# CLI
# @app.callback(invoke_without_command=True)
def add(
    name: str = typer.Argument(..., help="Name of new problem"),
    overwrite: bool = typer.Option(False,"--overwrite", help="Overwrite existing files"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show planned changes only"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompts"),
    force_git: bool = typer.Option(False,"--force-git", help="Automatically git add files"),
):
    """
    Generate a new PINN problem scaffold.
    """

    try:
        create_problem(
            name=name,
            overwrite=overwrite,
            dry_run=dry_run,
            yes=yes,
            force_git=force_git,
        )
    except Exception as e:
        typer.echo(f"[ERROR] {e}", err=True)
        raise typer.Exit(code=1)


# =========================================================
# DIRECT EXECUTION
# =========================================================
if __name__ == "__main__":
    app()