"""Basic example of coupling an external sampler with simulations."""

from optimas.core import VaryingParameter, Objective
from optimas.generators.base import Generator
from optimas.evaluators import TemplateEvaluator
from optimas.explorations import Exploration

from libensemble.gen_classes import UniformSample
import numpy as np


def analyze_simulation(simulation_directory, output_params):
    # Read back result from file
    with open("result.txt") as f:
        result = float(f.read())
    # Fill in output parameters.
    output_params["f"] = result
    return output_params


from optimas.generators.base import Generator


class StandardToTrials(Generator):

    def __init__(self, varying_parameters, objectives, ext_gen, **kwargs):
        super().__init__(varying_parameters, objectives, **kwargs)
        self.gen = ext_gen

    def ask(self, n_trials):
        return self.gen.ask(n_trials)

    def tell(self, trials):
        self.gen.tell(trials)


# Create varying parameters and objectives.
var_1 = VaryingParameter("x0", 0.0, 15.0, default_value=5.0)
var_2 = VaryingParameter("x1", 0.0, 15.0, default_value=6.0)
obj = Objective("f")

from libensemble.gen_classes import UniformSample
import numpy as np

libe_gen = UniformSample(
    gen_specs={
        "out": [("x", float, (2,))],
        "user": {
            "lb": np.array([0.0, 0.0]),
            "ub": np.array([15.0, 15.0]),
        },
    }
)


gen = StandardToTrials(
    varying_parameters=[var_1, var_2], objectives=[obj], ext_gen=libe_gen
)

# Create evaluator.
ev = TemplateEvaluator(
    sim_template="template_simulation_script.py",
    analysis_func=analyze_simulation,
)


# Create exploration.
exp = Exploration(
    generator=gen, evaluator=ev, max_evals=10, sim_workers=4, run_async=True
)


# To safely perform exploration, run it in the block below (this is needed
# for some flavours of multiprocessing, namely spawn and forkserver)
if __name__ == "__main__":
    exp.run()