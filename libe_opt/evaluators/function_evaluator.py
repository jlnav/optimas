from libe_opt.sim_functions import run_function
from .base import Evaluator


class FunctionEvaluator(Evaluator):
    def __init__(self, function, analyzed_params, n_gpus, n_proc):
        super().__init__()
        self.function = function
        self.analyzed_params = analyzed_params
        self.sim_function = run_function
        self.n_gpus = n_gpus
        self.n_proc = n_proc

    def get_sim_specs(self, variables, objectives):
        sim_specs = {
            # Function whose output is being minimized.
            'sim_f': self.sim_function,
            # Name of input for sim_f, that LibEnsemble is allowed to modify.
            # May be a 1D array.
            'in': [var.name for var in variables],
            'out': (
                [(obj.name, float) for obj in objectives]
                # f is the single float output that LibEnsemble minimizes.
                + [(par.name, par.type) for par in self.analyzed_params]
                # input parameters
                + [(var.name, float) for var in variables]
            ),
            'user': {
                'n_gpus': self.n_gpus,
                'n_proc': self.n_proc,
                'evaluation_func': self.function
            }
        }
        return sim_specs

    def get_libe_specs(self):
        libE_specs = {}
        return libE_specs
