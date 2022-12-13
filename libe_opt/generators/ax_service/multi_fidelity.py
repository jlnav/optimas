import torch

from ax.service.ax_client import AxClient
from ax.modelbridge.generation_strategy import (
    GenerationStep, GenerationStrategy)
from ax.modelbridge.registry import Models
from ax.service.utils.instantiation import ObjectiveProperties

from .base import AxServiceGenerator


class AxMultiFidelityGenerator(AxServiceGenerator):
    def __init__(self, variables, objectives=None, n_init=4, fidel_cost_intercept=1.):
        self.fidel_cost_intercept = fidel_cost_intercept
        super().__init__(variables, objectives, n_init)

    def _create_ax_client(self):
        # Create parameter list.
        parameters = list()
        for var in self.variables:
            parameters.append(
                {
                    'name': var.name,
                    'type': 'range',
                    'bounds': [var.lower_bound, var.upper_bound],
                    'is_fidelity': var.is_fidelity,
                    'target_value': var.target_value
                }
            )

        # Make generation strategy:
        steps = []

        # If there is no past history,
        # adds Sobol initialization with `batch_size` random trials:
        # if self.history is None:
        steps.append(
            GenerationStep(
                model=Models.SOBOL,
                num_trials=self.n_init
            )
        )

        # continue indefinitely with GPKG.
        steps.append(
                GenerationStep(
                    model=Models.GPKG,
                    num_trials=-1,
                    model_kwargs={
                        'cost_intercept': self.fidel_cost_intercept,
                        'torch_dtype': torch.double,
                        'torch_device': torch.device(self.torch_device)
                    }
                )
            )

        gs = GenerationStrategy(steps)

        ax_objectives = {}
        for obj in self.objectives:
            ax_objectives[obj.name] = ObjectiveProperties(minimize=obj.minimize)

        # Create client and experiment.
        ax_client = AxClient(generation_strategy=gs)
        ax_client.create_experiment(
            parameters=parameters,
            objectives=ax_objectives
        )

        self.ax_client = ax_client
