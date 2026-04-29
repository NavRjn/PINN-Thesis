import torch
import json
from pathlib import Path
from tqdm import trange
import importlib


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def training_loop(n_iters, problem_api, save_best_loss, on_train_end, logger=None):
    pbar = trange(n_iters)
    best_loss = float("inf")
    keys = problem_api.get_metric_keys()
    history = {'obj': {}, 'z': []} | dict.fromkeys(keys, {})
    problem = problem_api.problem

    model = problem.model
    optimizer = problem.optimizer
    loss_fn = problem.loss_fn

    try:
        print("Starting loop: Interrupting now will save something")
        for i in pbar:
            # logger.debug("Starting iteration {}/{}".format(i + 1, n_iters))
            optimizer.zero_grad()

            # Get training data/grid for this step
            batch = problem.grid_sampler()
            z = batch.get("z", None)

            # Calculate PDE Residual Loss
            loss, loss_metrics = loss_fn(model, batch)

            loss.backward()
            optimizer.step()

            # Logging
            pbar.set_description(f"Loss: {loss.item():.2e}")
            history['obj'][i] = loss_metrics.get("obj", 0)
            if z is not None: history['z'].extend(z.detach().cpu().view(-1).tolist())

            for key in keys:
                key_metric = loss_metrics.get(key, []) # default as list is perhaps a problem
                if key_metric is None:
                    logger.warn(f"key: {key} not found at in {key_metric} @ {i}")
                history[key][i] = key_metric

            if loss.item() < best_loss:
                best_loss = loss.item()
                save_best_loss(i, loss.item())

            # Checkpointing
            # if i % config["training"].get("save_freq", 1000) == 0:
            #     TODO: Implement other metrics: latent sensitivity, spectral metrics, etc. in the loss function and log them here
            #     torch.save(model.state_dict(), run_dir / "checkpoints" / f"model_{i}.pt")
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user at {i}. Saving current state.")
    except Exception as e:
        print("ERROR: ", e, e.with_traceback())
    finally:
        on_train_end(history)
        return history
        print("Training loop completed")


def get_problem(problem_name, config, logger):
    device = get_device()
    logger.info(f"Using device: {device}")
    problem_api = importlib.import_module(f"{problem_name}.api").API()
    problem_api.setup_problem(config, device, logger)
    return problem_api


def main(config, run_dir, logger):
    # 1. Setup Environment
    torch.manual_seed(config["training"].get("seed", 42))

    problem_api = get_problem(config['problem'], config, logger)
    problem = problem_api.problem
    model = problem.model

    # Metrics storage
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    # 3. Unified Training Loop
    n_iters = config["training"].get("n", 1_000)

    def save_best_loss(i, best_loss):
        # logger.debug(f"New best loss: {best_loss:.2e} at iteration {i + 1}")
        torch.save(model.state_dict(), run_dir / "checkpoints" / "best.pt")

    def on_train_end(history):
        torch.save(model.state_dict(), run_dir / "checkpoints" / "final.pt")
        with open(run_dir / "losses.json", "w") as f:
            json.dump(history, f)

        logger.info("Training completed. Running post-processing visualization.")
        problem_api.post_process(model, history, run_dir, problem.device)


    logger.info(f"Training run initiated: {run_dir}")

    _ = training_loop(
        n_iters=n_iters,
        problem_api=problem_api,
        save_best_loss=save_best_loss,
        on_train_end=on_train_end,
        logger=logger
                  )

