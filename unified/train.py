import torch
import json
from pathlib import Path
from tqdm import trange
import importlib


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(config, run_dir):
    # 1. Setup Environment
    device = get_device()
    torch.manual_seed(config.get("seed", 42))

    # 2. Dynamic Problem Loading
    # config["problem_name"] should be 'gray_scott' or 'bratu_1d'
    problem_name = config['problem']
    problem_module = importlib.import_module(f"{problem_name}.api")

    # We expect the problem module to provide a setup function
    model, optimizer, loss_fn, grid_sampler, logger = problem_module.setup_problem(config, device)

    # Metrics storage
    history = {'loss': []}
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    # 3. Unified Training Loop
    n_iters = config.get("n_iters", 1_000)
    pbar = trange(n_iters)

    for i in pbar:
        optimizer.zero_grad()

        # Get training data/grid for this step
        batch = grid_sampler()

        # Calculate PDE Residual Loss
        loss, loss_metrics = loss_fn(model, batch, config)

        loss.backward()
        optimizer.step()

        # Logging
        pbar.set_description(f"Loss: {loss.item():.2e}")
        history['loss'].append(loss.item())

        # Checkpointing
        if i % config.get("save_freq", 1000) == 0:
            torch.save(model.state_dict(), run_dir / "checkpoints" / f"model_{i}.pt")

    # 4. Finalize & Problem-specific Visualization
    torch.save(model.state_dict(), run_dir / "checkpoints" / "final.pt")
    with open(run_dir / "losses.json", "w") as f:
        json.dump(history, f)

    problem_module.post_process(model, history, run_dir, device)