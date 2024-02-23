from typing import Dict
import copy
from gymnasium import Env
import logging

from torch.nn.modules import Module
from flwr.common.logger import logger
import torch

import curiosity
from curiosity.experience import Transition
from curiosity.rl.dqn import DQN

from common.interface import GymnasiumActorClient

# https://arxiv.org/pdf/1507.04296.pdf
# Massively Parallel Methods for Deep Reinforcement Learning
# Nair et al.

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLIENTS = 10

# TODO: Maybe move all algorithms into library
# TODO: Separate config out, maybe with Hydra
# TODO: Typed Configuration
# TODO: Generalised curiosity client, to use any algorithm implemented by curiosity

class DQNClient(GymnasiumActorClient):
    def __init__(self, cid: int, env: Env, net: Module, config: Dict):
        self.cid = cid
        logger.log(
            logging.INFO,
            f"Creating DQN client with cid {cid}"
        )

            # Copy, since we need to pop some keys from config
        config = copy.deepcopy(config)
        config.get("algorithm", {}).pop("critic", None)
        super().__init__(cid, env, net, config)

            # TODO: Improve resource management
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        net = net.to(self.device)

            # Logging
        self.evaluator = curiosity.logging.CuriosityEvaluator(
            env,
            device=self.device,
            **self.cfg.get("evaluation", {})
        )

            # Initialise Reinforcement Learning Modules
        self.memory = curiosity.experience.util.build_replay_buffer(
            env=env,
            device=self.device,
            **self.cfg.get("memory", {})
        )
        self.algorithm = DQN(
            critic=self.net,
            **self.cfg.get("algorithm", {}),
            device=self.device
        )
        self.policy = curiosity.policy.EpsilonGreedyPolicy(
            fn=self.algorithm.policy_fn,
            action_space= self.env.action_space,
            device=self.device
        )
        self.evaluator.policy = self.policy
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
        metrics = {}
        # Synchronise critic net
        self.algorithm.critic.net = net

        # Training
        for _ in range(train_config["frames"]):
            self._step += 1
            # Collected Transitions
            self.collector.collect(n=1)
            batch, aux = self.memory.sample(self.cfg["train"]["minibatch_size"])
            batch = Transition(*batch)
            # Algorithm Update
            critic_loss = self.algorithm.update(batch, aux, self.step)

        # Logging
        metrics["loss"] = critic_loss
        return metrics, train_config["frames"]

    def evaluate(self, parameters, config: Dict):
        super().evaluate(parameters, config)
        repeats = config.get("evaluation_repeats", self.evaluator.repeats)
        reward = self.evaluator.evaluate(repeats=repeats)
        # TODO: What is loss under this framework? Should be removed by API
        return 1.0, repeats, {"reward": reward}

    @property
    def step(self) -> int:
        return self._step
    
    # TODO: How to close resources?

def create_dqn_client(cid: int, config: Dict) -> DQNClient:
    # Build env
        # Construction
    env = curiosity.util.build_env(**config["rl"]["env"])

    # Build net
    net = curiosity.util.build_critic(
        env=env,
        features=config["rl"]["algorithm"]["critic"]["features"]
    )

    return DQNClient(cid, env, net, config["rl"])
