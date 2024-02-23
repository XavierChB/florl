from typing import Dict
import copy
from gymnasium import Env

from torch.nn.modules import Module
import flwr as fl
import torch
from omegaconf import OmegaConf, DictConfig

import curiosity
from curiosity.experience import Transition
from curiosity.rl.dqn import DQN

from common.interface import GymnasiumActorClient

# https://arxiv.org/pdf/1507.04296.pdf
# Massively Parallel Methods for Deep Reinforcement Learning
# Nair et al.

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLIENTS = 10

# TODO: Evaluation
# TODO: Separate config out, maybe with Hydra
# TODO: Typed Configuration
# TODO: Generalised curiosity client, to use any algorithm implemented by curiosity

class DQNClient(GymnasiumActorClient):
    def __init__(self, env: Env, net: Module, config: Dict):
            # Copy, since we need to pop some keys from config
        config = copy.deepcopy(config)
        config["algorithm"].pop("critic", None)
        super().__init__(env, net, config)

            # TODO: Improve resource management
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = net.to(self.device)

            # Logging
        self.evaluator = curiosity.logging.CuriosityEvaluator(
            env,
            video=False,
            device=self.device,
            **self.cfg["evaluation"]
        )

            # Initialise Reinforcement Learning Modules
        self.memory = curiosity.experience.util.build_replay_buffer(
            env=env,
            device=self.device,
            **self.cfg["memory"]
        )
        self.algorithm = DQN(
            critic=self.net,
            **self.cfg["algorithm"],
            device=self.device
        )
        self.policy = curiosity.policy.EpsilonGreedyPolicy(
            fn=self.algorithm.policy_fn,
            action_space= self.env.action_space,
            device=self.device
        )
        self.collector = curiosity.experience.util.build_collector(
            policy=self.policy,
            env = self.env,
            memory=self.memory,
            device=self.device
        )
            # Early Start
        self._step = 0
        self.collector.early_start(n=self.cfg["train"]["initial_collection_size"])

    def train(self, net: Module, train_config: Dict):
        # Synchronise critic net
        self.algorithm.critic.net = net

        # Training
        for i  in range(train_config["frames"]):
            self._step += 1
            # Collected Transitions
            self.collector.collect(n=1)
            batch, aux = self.memory.sample(self.cfg["train"]["minibatch_size"])
            batch = Transition(*batch)
            # Algorithm Update
            self.algorithm.update(batch, aux, self.step)

        return {}, train_config["frames"]

    def evaluate(self, parameters, config: Dict):
        super().evaluate(parameters, config)


    @property
    def step(self) -> int:
        return self._step
    
    # TODO: How to close resources?

def create_dqn_client(cid: int, config: Dict) -> DQNClient:
    # Build env
        # Construction
    env = curiosity.util.build_env(**config["rl"]["env"])
        # Seeding
    env.reset(seed=cid)
    env.action_space.seed(cid)

    # Build net
    net = curiosity.util.build_critic(
        env=env,
        features=config["rl"]["algorithm"]["critic"]["features"]
    )

    return DQNClient(env, net, config["rl"])

config = DictConfig({
    "rl": {
        "env": {
            "name": "CartPole-v1"
        },
        "algorithm": {
            "type": "ddpg",
            "gamma": 0.99,
            "tau": 0.005,
            "lr": 0.001,
            "update_frequency": 1,
            "clip_grad_norm": 1,
            "critic": {
                "features": 64
            }
        },
        "memory": {
            "type": "experience_replay",
            "capacity": 20000
        },
        "train": {
            "initial_collection_size": 512,
            "minibatch_size": 32
        }
    },
    "fl": {
        "train_config": {
            "frames": 100,
        }
    }
})

# Gorila with Flower
train_config = OmegaConf.to_container(config["fl"]["train_config"])
def _on_fit_config_fn(server_round: int):
    return train_config | {"server_round": server_round}

strategy = fl.server.strategy.FedAvg(
    on_fit_config_fn = _on_fit_config_fn
)

fl.simulation.start_simulation(
    client_fn=lambda cid: create_dqn_client(int(cid), config=config).to_client(),
    client_resources={'num_cpus': 1},
    config=fl.server.ServerConfig(num_rounds=10),
    num_clients = NUM_CLIENTS,
    strategy = strategy
)
