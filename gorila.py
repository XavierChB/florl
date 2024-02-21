from typing import Dict

from torch.nn.modules import Module
import flwr as fl
import torch
from omegaconf import DictConfig
from curiosity.util import build_env, build_critic

from common.interface import GymnasiumActorClient

# https://arxiv.org/pdf/1507.04296.pdf
# Massively Parallel Methods for Deep Reinforcement Learning
# Nair et al.

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DQNClient(GymnasiumActorClient):
    def train(net: Module, config: Dict):
        return super().train(config)

def create_dqn_client(cid: int, config) -> DQNClient:
    # Build env
        # Construction
    env = build_env(**config["env"])
        # Seeding
    env.reset(seed=cid)
    env.action_space.seed(cid)

    # Build net
    net = build_critic(env, features = config["critic"]["features"])

    return DQNClient(env, net)

# TODO: Separate config out, maybe with Hydra
config = {
    "env": {
        "name": "CartPole-v1"
    },
    "critic": {
        "features": 64
    }
}

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0, # All actors available
)
create_dqn_client(0, config)