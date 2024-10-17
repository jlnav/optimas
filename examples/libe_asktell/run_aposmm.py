"""Basic example of coupling an external sampler with simulations."""

from optimas.core import VaryingParameter, Objective, TrialParameter
from optimas.generators.base import Generator
from optimas.evaluators import TemplateEvaluator
from optimas.explorations import Exploration
from libensemble.gen_classes import UniformSample
import numpy as np

import multiprocessing

multiprocessing.set_start_method("fork", force=True)

from math import gamma, pi, sqrt

import libensemble.gen_funcs

libensemble.gen_funcs.rc.aposmm_optimizers = "nlopt"
from libensemble.gen_classes import APOSMM
from libensemble.specs import AllocSpecs, ExitCriteria, GenSpecs, SimSpecs
from libensemble.tests.regression_tests.support import (
    six_hump_camel_minima as minima,
)
from libensemble.tools import add_unique_random_streams


def analyze_simulation(simulation_directory, output_params):
    # Read back result from file
    with open("result.txt") as f:
        result = float(f.read())
    # Fill in output parameters.
    output_params["f"] = result
    return output_params


from optimas.generators.base import Generator
import numpy as np


class StandardToTrials(Generator):

    def __init__(
        self,
        varying_parameters,
        objectives,
        custom_trial_parameters,
        ext_gen,
        **kwargs,
    ):
        super().__init__(
            varying_parameters,
            objectives,
            custom_trial_parameters=custom_trial_parameters,
            **kwargs,
        )
        self.gen = ext_gen
        self.gen.setup()
        self.queued_ask = None

    def ask_trials(self, n_trials):
        trials = super().ask_trials(n_trials)
        for i, trial in enumerate(trials):
            trial.local_min = self.queued_ask[i]["local_min"]
            trial.local_pt = self.queued_ask[i]["local_pt"]
            trial.sim_id = self.queued_ask[i]["sim_id"]
        return trials

    def ask(self, n_trials):
        self.queued_ask = self.gen.ask(n_trials)
        return self.queued_ask

    def tell(self, trials):
        for trial in trials:  # convert output tuple of (fval, None) to fval
            trial["f"] = trial["f"][0]
            trial.pop("_id")
            trial.pop("_ignored")
            trial.pop("_ignored_reason")
            trial.pop("_status")
        self.gen.tell(trials)

# Create varying parameters and objectives.
n = 2
var_1 = VaryingParameter("x0", -3.0, 3.0)
var_2 = VaryingParameter("x1", -2.0, 2.0)
var_3 = VaryingParameter("x_on_cube0", 0, 1.0)
var_4 = VaryingParameter("x_on_cube1", 0, 1.0)

local_min = TrialParameter("local_min", save_name="local_min", dtype=bool)
local_pt = TrialParameter("local_pt", save_name="local_pt", dtype=bool)
sim_id = TrialParameter("sim_id", save_name="sim_id", dtype=int)

obj = Objective("f")

persis_info = add_unique_random_streams({}, 5)[1]
persis_info["nworkers"] = 4

aposmm = APOSMM(
    persis_info=persis_info,
    initial_sample_size=100,
    sample_points=np.round(minima, 1),
    localopt_method="LN_BOBYQA",
    rk_const=0.5 * ((gamma(1  (n / 2)) * 5) ** (1 / n)) / sqrt(pi),
    xtol_abs=1e-4,
    ftol_abs=1e-4,
    max_active_runs=4,
    dist_to_bound_multiple=0.5,
    lb=np.array([var_1.lower_bound, var_2.lower_bound]),
    ub=np.array([var_1.upper_bound, var_2.upper_bound]),
)

aposmm.setup()

gen = StandardToTrials(
    varying_parameters=[var_1, var_2, var_3, var_4],
    objectives=[obj],
    custom_trial_parameters=[local_min, local_pt, sim_id],
    ext_gen=aposmm,
)

# Create evaluator.
ev = TemplateEvaluator(
    sim_template="template_simulation_script.py",
    analysis_func=analyze_simulation,
)


# Create exploration.
exp = Exploration(
    generator=gen, evaluator=ev, max_evals=1000, sim_workers=4, run_async=True
)


# To safely perform exploration, run it in the block below (this is needed
# for some flavours of multiprocessing, namely spawn and forkserver)
if __name__ == "__main__":
    exp.run()
    H, _, _ = aposmm.final_tell()
    print(H[H["local_min"]])