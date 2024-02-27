from abc import ABC

import gymnasium as gym
import flwr as fl
from flwr.common.typing import (
    GetParametersIns,
    GetParametersRes,
    Config,
    Status,
    Code,
    Parameters,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
)

from florl.common.knowledge import Knowledge


class RLClient(fl.client.Client, ABC):
    """A client interface specific to reinforcement learning"""

    def __init__(self, knowledge: Knowledge):
        self._knowl = knowledge

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return self._knowl.get_parameters(ins)

    def fit(self, ins: FitIns) -> FitRes:
        self._knowl.set_parameters(ins.parameters)
        return self.train(ins.config)

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        self._knowl.set_parameters(ins.parameters)
        return self.epoch(ins.config)

    # =========
    # Overide these two methods
    # =========

    def train(self, config: Config) -> FitRes:
        """Runs a reinforcement learning algorithm to update local knowledge

        Args:
            config: Training Configuration
        """
        return FitRes(
            status=Status(
                code=Code.FIT_NOT_IMPLEMENTED,
                message="Client does not implement `train`",
            ),
            parameters=Parameters(tensor_type="", tensors=[]),
            num_examples=0,
            metrics={},
        )

    def epoch(self, config: Config) -> EvaluateRes:
        """An epoch represents a evaluation run

        Args:
            config (Config): Evaluation configuration
        """
        return EvaluateRes(
            status=Status(
                code=Code.EVALUATE_NOT_IMPLEMENTED,
                message="Client does not implement `epoch`",
            ),
            loss=0.0,
            num_examples=0,
            metrics={},
        )


class GymClient(RLClient, ABC):
    """An RL client training within a gym environment"""

    def __init__(self, knowledge: Knowledge, env: gym.Env, seed: int | None = None):
        super().__init__(knowledge)
        self._env = env
        self._seed = seed

        self._env.reset(seed=self._seed)
