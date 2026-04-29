
class BaseProblemAPI:
    def __init__(self):
        self.metric_keys = []

        self.model = None
        self.optimizer = None
        self.loss_fn = None
        self.grid_sampler = None
        self.logger = None
        self.device = None

    def get_metric_keys(self):
        return self.metric_keys

    def _init_problem(self, model, optimizer, loss_fn, grid_sampler, logger, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.grid_sampler = grid_sampler
        self.logger = logger
        self.device = device

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

    def post_process(self, history, run_dir):
        """
        This method can be optionally implemented by each problem-specific API for any post-training processing or visualization.
        By default, it does nothing.
        """
        pass

    @classmethod
    def post_process_visualize(cls, run_dir, config):
        """
        This method can be optionally implemented by each problem-specific API for any additional visualization after training.
        By default, it does nothing.
        """
        pass