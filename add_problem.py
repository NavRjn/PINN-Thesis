import argparse
from pathlib import Path
import shutil
import sys


TEMPLATE_DIR = Path("templates/zero")
PROBLEMS_DIR = Path() # Root directory for now
CONFIGS_DIR = Path("configs")


# --------------------------------------------------
# UTIL
# --------------------------------------------------

def render_template(path: Path, replacements: dict) -> str:
    txt = path.read_text()
    for k, v in replacements.items():
        txt = txt.replace(f"{{{{ {k} }}}}", v)
    return txt


def file_plan(name: str):
    """Compute what will be created/overwritten"""

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
    print("\n=== ACTION PLAN ===")
    for typ, path, exists in plan:
        status = "OVERWRITE ⚠" if exists else "CREATE"
        print(f"{typ:6} | {status:10} | {path}")
    print("===================\n")


def confirm(prompt: str):
    ans = input(prompt + " [y/N]: ").strip().lower()
    return ans == "y"


def git_add(files):
    import subprocess

    paths = [str(f) for f in files]

    subprocess.run(["git", "add", *paths], check=False)


def print_next_steps(problem_dir: Path, config_path: Path):
    api_path = problem_dir / "api.py"
    problem_path = problem_dir / "problem.py"
    readme_path = problem_dir / "README.md"

    print("\n" + "="*60)
    print("🚀 Problem scaffold created successfully!")
    print("="*60)

    print("\n📌 NEXT STEPS (minimal guide)\n")

    print("1) Configure experiment:")
    print(f"   👉 {config_path}")
    print("   - change model, training, physics parameters\n")

    print("2) Define problem logic:")
    print(f"   👉 {problem_path}")
    print("   - implement grid_sampler() if needed")
    print("   - implement loss_fn(model, batch)\n")

    print("3) (Optional) adjust API behavior:")
    print(f"   👉 {api_path}")
    print("   - only if you need custom model/optimizer logic\n")

    print("4) Full documentation:")
    print(f"   👉 {readme_path}\n")

    print("💡 Tip:")
    print("   Start by running training immediately.")
    print("   Modify ONLY problem.py first unless necessary.\n")

    print("="*60 + "\n")


# --------------------------------------------------
# CORE
# --------------------------------------------------

def create_problem(name: str, overwrite: bool, dry_run: bool, force_git: bool):

    plan = file_plan(name)
    print_plan(plan)

    created_files = []

    dangerous = any(exists for _, _, exists in plan)

    if dry_run:
        print("[DRY RUN] No files written.")
        print_plan(plan)
        return

    print_plan(plan)

    # ALWAYS confirm before any write
    if not confirm("Proceed with creation?"):
        print("Aborted.")
        return

    problem_dir = PROBLEMS_DIR / name
    problem_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # CONFIG
    # --------------------------------------------------

    base_cfg = CONFIGS_DIR / "generic.yaml"
    cfg_text = base_cfg.read_text()

    cfg_text = cfg_text.replace(
        'problem: "generic_problem"',
        f'problem: "{name}"'
    )

    cfg_path = CONFIGS_DIR / f"{name}.yaml"

    if cfg_path.exists() and not overwrite:
        print(f"[SKIP] {cfg_path}")
    else:
        cfg_path.write_text(cfg_text)
        created_files.append(cfg_path)
        print(f"[OK] {cfg_path}")

    # --------------------------------------------------
    # PY FILES
    # --------------------------------------------------

    replacements = {"problem_name": name}

    for tpl in TEMPLATE_DIR.glob("*.tpl"):

        if tpl.name == "README.tpl":
            out_path = problem_dir / "README.md"
        else:
            out_path = problem_dir / (tpl.stem + ".py")

        if out_path.exists() and not overwrite:
            print(f"[SKIP] {out_path}")
            continue

        content = render_template(tpl, replacements)

        out_path.write_text(content)
        created_files.append(out_path)

        print(f"[OK] {out_path}")

    # --------------------------------------------------
    # INIT
    # --------------------------------------------------

    init_file = problem_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")
        created_files.append(init_file)

    if force_git or confirm("Add generated files to git?"):
            git_add(created_files)
            print("[OK] added to git")

    print_next_steps(problem_dir, cfg_path)


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate new PINN problem scaffold")

    parser.add_argument("name")
    parser.add_argument("--overwrite", action="store_true",help="Overwrite existing files")
    parser.add_argument("--dry-run",action="store_true",help="Show what would be created")
    parser.add_argument("--yes", action="store_true", help="skip confirmation")
    parser.add_argument("--open", action="store_true", help="open key files after creation")
    parser.add_argument("--force-git", action="store_true", help="git add generated files")

    args = parser.parse_args()

    create_problem(args.name, args.overwrite, args.dry_run, args.force_git)


if __name__ == "__main__":
    main()