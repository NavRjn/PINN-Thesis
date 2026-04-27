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
    torch.manual_seed(config["training"].get("seed", 42))


    # 2. Dynamic Problem Loading
    # config["problem_name"] should be 'gray_scott' or 'bratu_1d'
    problem_name = config['problem']
    problem_module = importlib.import_module(f"{problem_name}.api")

    # We expect the problem module to provide a setup function
    model, optimizer, loss_fn, grid_sampler, logger = problem_module.setup_problem(config, device)

    # Metrics storage
    history = {'obj': {}, 'grad': {}, 'div': {}, 'residual': {}, 'latent_sensitivity': {}, 'spectral': {}}
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    best_loss = float('inf')
    z_history = []

    # 3. Unified Training Loop
    n_iters = config["training"].get("n", 1_000)
    pbar = trange(n_iters)


    try:
        print("Starting loop: Interrupting now will save something")
        for i in pbar:
            optimizer.zero_grad()

            # Get training data/grid for this step
            batch = grid_sampler()
            z = batch["z"]

            # Calculate PDE Residual Loss
            loss, loss_metrics = loss_fn(model, batch)

            loss.backward()
            optimizer.step()

            # Logging
            pbar.set_description(f"Loss: {loss.item():.2e}")
            history['obj'][i] = loss_metrics["obj"]
            history["grad"][i] = loss_metrics["grad"]
            z_history.extend(z.detach().cpu().view(-1).tolist())

            if loss.item() < best_loss:
                best_loss = loss.item()
                logger.debug(f"New best loss: {best_loss:.2e} at iteration {i+1}")
                torch.save(model.state_dict(), run_dir / "checkpoints" / "best.pt")

            # Checkpointing
            # if i % config["training"].get("save_freq", 1000) == 0:
            #     TODO: Implement other metrics: latent sensitivity, spectral metrics, etc. in the loss function and log them here
            #     torch.save(model.state_dict(), run_dir / "checkpoints" / f"model_{i}.pt")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving current state.")
        print("\nTraining interrupted by user. Saving current state.")
    finally:
        # 4. Finalize & Problem-specific Visualization
        torch.save(model.state_dict(), run_dir / "checkpoints" / "final.pt")
        with open(run_dir / "losses.json", "w") as f:
            json.dump(history, f)

        problem_module.post_process(model, history, z_history, run_dir, device)