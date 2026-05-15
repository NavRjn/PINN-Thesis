from pathlib import Path
import matplotlib.pyplot as plt

def plot_loss_curves(history, run_dir):
    run_dir = Path(run_dir)
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    iters = sorted(history["obj"].keys())
    loss = [history["obj"][i] for i in iters]

    plt.plot(iters, loss)
    plt.yscale("log")
    plt.xlabel("iter")
    plt.ylabel("loss")
    plt.title("training")
    plt.savefig(fig_dir / "loss.png", dpi=150)
    plt.close()