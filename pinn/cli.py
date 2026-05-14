import typer
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pinn.commands.train import app as train_app
from pinn.commands.visualize import app as viz_app
from pinn.commands.add import app as add_app

app = typer.Typer()

app.add_typer(train_app, name="train")
app.add_typer(viz_app, name="viz")
app.add_typer(add_app, name="add")

if __name__ == "__main__":
    app()