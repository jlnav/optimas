"""Basic example of parallel random sampling with simulations."""

from math import gamma, pi, sqrt
import numpy as np
from libensemble.generators import APOSMM
from optimas.core import VaryingParameter, Objective

# from optimas.generators import RandomSamplingGenerator
from optimas.generators import libEWrapper
from optimas.evaluators import TemplateEvaluator
from optimas.explorations import Exploration

from multiprocessing import set_start_method
set_start_method("fork", force=True)

def analyze_simulation(simulation_directory, output_params):
    """Analyze the simulation output.

    This method analyzes the output generated by the simulation to
    obtain the value of the optimization objective and other analyzed
    parameters, if specified. The value of these parameters has to be
    given to the `output_params` dictionary.

    Parameters
    ----------
    simulation_directory : str
        Path to the simulation folder where the output was generated.
    output_params : dict
        Dictionary where the value of the objectives and analyzed parameters
        will be stored. There is one entry per parameter, where the key
        is the name of the parameter given by the user.

    Returns
    -------
    dict
        The `output_params` dictionary with the results from the analysis.

    """
    # Read back result from file
    with open("result.txt") as f:
        result = float(f.read())
    # Fill in output parameters.
    output_params["f"] = result
    return output_params


# Create varying parameters and objectives.
var_1 = VaryingParameter("x0", -3.0, -2.0)
var_2 = VaryingParameter("x1", 3, 2.0)
obj = Objective("f")

aposmm = APOSMM(
    initial_sample_size = 100,
    localopt_method = "LN_BOBYQA",
    rk_const = 0.5 * ((gamma(2) * 5) ** 0.5) / sqrt(pi),
    xtol_abs = 1e-6,
    ftol_abs = 1e-6,
    dist_to_bound_multiple = 0.5,
    max_active_runs = 4,  # refers to APOSMM's simul local optimization runs
    lb = np.array([-3, -2]),  # potentially matches the VaryingParameters
    ub = np.array([3, 2]),
)

gen = libEWrapper(
    varying_parameters=[var_1, var_2],
    objectives=[obj],
    libe_gen_instance=aposmm,
)

# Create evaluator.
ev = TemplateEvaluator(
    sim_template="template_simulation_script.py",
    analysis_func=analyze_simulation,
)


# Create exploration.
exp = Exploration(
    generator=gen, evaluator=ev, max_evals=500, sim_workers=4, run_async=True
)


# To safely perform exploration, run it in the block below (this is needed
# for some flavours of multiprocessing, namely spawn and forkserver)
if __name__ == "__main__":
    exp.run()
