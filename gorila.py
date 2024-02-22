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
NUM_CLIENTS = 10

class DQNClient(GymnasiumActorClient):
    def train(self, net: Module, config: Dict):
        # TODO
        return {}, 1

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
config = DictConfig({
    "env": {
        "name": "CartPole-v1"
    },
    "critic": {
        "features": 64
    }
})

# Gorila with Flower
strategy = fl.server.strategy.FedAvg()
fl.simulation.start_simulation(
    client_fn=lambda cid: create_dqn_client(int(cid), config=config).to_client(),
    client_resources={'num_cpus': 1},
    config=fl.server.ServerConfig(num_rounds=1),
    num_clients = NUM_CLIENTS,
    strategy = strategy
)
