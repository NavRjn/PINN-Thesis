from core.utils import ProblemSetup


class BaseProblemAPI:
    def __init__(self):
        self.problem: ProblemSetup = None
        self.metric_keys = []

    def get_metric_keys(self):
        return self.metric_keys

    def setup_problem(self, config, device, logger):
        """
        This method should be implemented by each problem-specific API.
        It should return a ProblemSetup object containing:
        - model: The neural network architecture for this problem
        - optimizer: The optimizer for training
        - loss_fn: A function that computes the PDE residual loss given the model and input batch
        - grid_sampler: A function that generates the training data/grid for each iteration
        - logger: A logging object for recording metrics and checkpoints
        """
        raise NotImplementedError("Each problem API must implement the setup_problem method.")

    def post_process(self, model, history, z_history, run_dir, device):
        """
        This method can be optionally implemented by each problem-specific API for any post-training processing or visualization.
        By default, it does nothing.
        """
        pass

    @classmethod
    def post_process_visualize(cls, run_dir, config, device):
        """
        This method can be optionally implemented by each problem-specific API for any additional visualization after training.
        By default, it does nothing.
        """
        pass