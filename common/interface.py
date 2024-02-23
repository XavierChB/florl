from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, OrderedDict, Tuple
import numbers
import logging


from flwr.common import NDArrays
import numpy as np

import gymnasium as gym
import flwr as fl
import torch
from torch import nn

# TODO: Design improved interface for generalised synchronous FRL
# - How tightly should this be linked with existing libraries and requirements?
# - Enforce the curiosity library as a requirements? May be good for evaluation.
# - Packaging and integration as a PR into flwr

# ==========
# Functions from https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html and the labs

# TODO: move these into the API somehow
def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def aggregate_weighted_average(metrics: List[Tuple[int, dict]]) -> dict:
    """Generic function to combine results from multiple clients
    following training or evaluation.

    Args:
        metrics (List[Tuple[int, dict]]): collected clients metrics

    Returns:
        dict: result dictionary containing the aggregate of the metrics passed.
    """
    average_dict: dict = defaultdict(list)
    total_examples: int = 0
    for num_examples, metrics_dict in metrics:
        for key, val in metrics_dict.items():
            if isinstance(val, numbers.Number):
                average_dict[key].append((num_examples, val))  # type:ignore
        total_examples += num_examples
    return {
        key: {
            "avg": float(
                sum([num_examples * metr for num_examples, metr in val])
                / float(total_examples)
            ),
            "all": val,
        }
        for key, val in average_dict.items()
    }

# ==========

class GymnasiumActorClient(fl.client.NumPyClient, ABC):
    """  A client for federated reinforcement learning
    """

    def __init__(self, cid: int, env: gym.Env, net: nn.Module, config: Dict):
        """ Initialises the client

        Args:
            env (gym.Env): gymnasium environment to explore in.
            net (nn.Module): nn module to as the target for federated training. Can be a policy network, world module, Q-estimator etc.
        """
        self.cid = cid
        self.env = env
        self.net = net
        self.cfg = config

    def fit(self, parameters, config: Dict):
        set_parameters(self.net, parameters)
        metrics, n_examples = self.train(self.net, config)
        return self.get_parameters(config=config), n_examples, metrics
    
    def evaluate(self, parameters, config: Dict):
        set_parameters(self.net, parameters)
        return 0.0, 1, {}

    def get_parameters(self, config: Dict) -> NDArrays:
        return get_parameters(self.net)

    @abstractmethod
    def train(self, net: nn.Module, config: Dict) -> Tuple[Dict, int]:
        """ Runs a reinforcement learning algorithm to update net

        Args:
            net (nn.Module): net.
            config (Dict): configuration file.

        Returns:
            Dict: Fit metrics
            int:  Recommended weighting factor (corresponds to number of examples), e.g replay buffer size or collected frames.
        """
        raise NotImplementedError
