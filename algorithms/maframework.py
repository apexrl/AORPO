import argparse
import copy
from typing import Dict, List, Tuple

import numpy as np
import torch
from agents import *
from agents.baseagent import AgentMB
from gym.spaces import Discrete
from torch.utils.tensorboard import SummaryWriter
from utils.buffer import ReplayBuffer

from .baseframework import BaseFramework

NLLLoss = torch.nn.NLLLoss()


class MA_Controller(BaseFramework):
    def __init__(
        self,
        agent_init_params: List[Dict],
        alg_types: List[str],
        rew_scale: float = 1.0,
        gamma: float = 0.95,
        tau: float = 0.01,
        lr: float = 0.01,
        hidden_dim: int = 64,
        discrete_action: int = True,
    ):
        super().__init__(
            alg_types,
            rew_scale=rew_scale,
            gamma=gamma,
            tau=tau,
            lr=lr,
            hidden_dim=hidden_dim,
            discrete_action=discrete_action,
        )
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                (possible argments)
                dim_in_pol (int): Input dimensions to policy
                dim_out_pol (int): Output dimensions to policy
                dim_in_critic (int): Input dimensions to critic
                dim_in_opp_pols (int): Input dimensions to opponent policies
                dim_out_opp_pols (int): Output dimensions to opponent policies
                n_agent (int): number of agents
                agent_index (int): index of the agent
                alg_type (str): name of the algorithm the agent uses
                auto_target_entropy (bool): learn the temperature automatically
                target_entropy (numeral or function)
                reparameterize (bool): seems useless
                action_shape_list (bool): in some environment the agents can do many actions at a time

            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.agent_init_params = agent_init_params
        self.alg_types = alg_types
        self.agents: List[BaseAgent] = []
        self.alg_names = [
            "DDPG",
            "MADDPGOppMd",
            "AORDPG",
            "SAC",
            "MASACOppMd",
            "AORPO",
        ]
        self.name_to_alg = dict(
            zip(
                self.alg_names,
                [
                    DDPGAgent,
                    DDPGAgentOppMd,
                    DDPGAgentOppMdCondMB,
                    SACAgent,
                    SACAgentOppMd,
                    SACAgentOppMdCondMB,
                ],
            )
        )
        for i, params in enumerate(agent_init_params):
            print(params)
            self.agents.append(
                self.name_to_alg[alg_types[i]](
                    lr=lr,
                    hidden_dim=hidden_dim,
                    rew_scale=rew_scale,
                    discrete_action=discrete_action,
                    **params
                )
            )

        self.total_difficulty = 0
        self.agent_difficulties = []
        for i, a in enumerate(self.agents):
            print("agent", i, a.alg_type)
            self.total_difficulty += a.dim_in_pol + a.dim_out_pol
            self.agent_difficulties.append(a.dim_in_pol + a.dim_out_pol)

    def random_step(self, observations: List):
        return [a.random_step(obs) for a, obs in zip(self.agents, observations)]

    def step(self, observations: List, explore: bool = False, return_raw: bool = False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
            return_raw(boolean): Whether return the raw action
        Outputs:
            actions: List of actions for each agent
        """
        if return_raw:
            # for discrete actions, return probabilities
            actions = [
                a.step(observations, explore=explore, return_raw=True)
                if (isinstance(a, AgentCond))
                else a.step(obs, explore=explore, return_raw=True)
                for a, obs in zip(self.agents, observations)
            ]
            actions = torch.stack(actions)
            actions = torch.softmax(actions, dim=2)
            return actions.detach()
        else:
            return [
                a.step(observations, explore=explore)
                if isinstance(a, AgentCond)
                else a.step(obs, explore=explore)
                for a, obs in zip(self.agents, observations)
            ]

    def update(
        self,
        sample: Tuple,
        agent_i: int,
        logger: SummaryWriter = None,
        logger_iter: int = 0,
    ):
        """
        Update parameters of a agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        loss_dict = self.agents[agent_i].update(sample)
        if logger and loss_dict:
            logger.add_scalars("agent%i/losses" % (agent_i), loss_dict, logger_iter)

    def get_dynamics_model(self, agent_i: int) -> Dict:
        assert isinstance(self.agents[agent_i], AgentMB), "get dynamics model"
        return copy.deepcopy(self.agents[agent_i].dynamics_model.state_dict())

    def set_dynamics_model(self, agent_i: int, model: Dict):
        assert isinstance(self.agents[agent_i], AgentMB), "get dynamics model"
        self.agents[agent_i].dynamics_model.load_state_dict(model)

    def get_opponent_model(self, agent_i: int) -> List[Dict]:
        assert isinstance(self.agents[agent_i], AgentOppMd), "get opponent model"
        return [
            copy.deepcopy(opp_pl.state_dict())
            for opp_pl in self.agents[agent_i].opp_policies
        ]

    def set_opponent_model(self, agent_i: int, models: List[Dict]):
        assert isinstance(self.agents[agent_i], AgentOppMd), "get opponent model"
        for k, opp_pl in enumerate(self.agents[agent_i].opp_policies):
            opp_pl.load_state_dict(models[k])

    def update_opponent_models(
        self,
        sample: List,
        logger: SummaryWriter = None,
        logger_iter: int = 0,
        epochs: int = 1,
    ):
        """
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled the
                    latest data from the replay buffer.
                    Each is a list with entries corresponding to each agent.
        """
        for i, a in enumerate(self.agents):
            for e in range(epochs):
                if isinstance(a, AgentOppMd):
                    opp_loss = a.update_opponent(sample)

                    if e == epochs - 1 and logger and opp_loss:
                        logger.add_scalars(
                            "agent%i/model_losses" % (i), opp_loss, logger_iter
                        )

    def rollout_models(
        self,
        K: int,
        init_states: List,
        logger: SummaryWriter = None,
        logger_iter: int = 0,
    ):
        for a in self.agents:
            if isinstance(a, AgentMB):
                a.rollout_model(K, init_states, self.agents, logger, logger_iter)

    def update_dynamics_models(
        self,
        sample: Tuple[List],
        logger: SummaryWriter = None,
        logger_iter: int = 0,
        epochs: int = 1,
    ):
        for i, a in enumerate(self.agents):
            if isinstance(a, AgentMB):
                m_loss_dict = a.update_model(sample, epochs)
                if logger and m_loss_dict:
                    logger.add_scalars(
                        "agent%i/model_losses" % (i), m_loss_dict, logger_iter
                    )

    def save(self, model_path: str):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device="cpu")  # move parameters to CPU before saving
        save_dict = {
            "init_dict": self.init_dict,
            "agent_params": [a.get_params() for a in self.agents],
        }
        torch.save(save_dict, model_path)

    @classmethod
    def init_from_save(cls, model_path: str):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(model_path, map_location=torch.device("cpu"))
        instance = cls(**save_dict["init_dict"])
        instance.init_dict = save_dict["init_dict"]
        for a, params in zip(instance.agents, save_dict["agent_params"]):
            a.load_params(params)
        return instance

    def update_all_targets(self):
        for a in self.agents:
            if type(a) != BaseAgent:
                a.update_targets(self.tau)

    def prep_rollouts(self, device="cpu"):
        for a in self.agents:
            a.prep_rollouts(device)

    def prep_training(self, device="cuda"):
        for a in self.agents:
            a.prep_training(device)

    def compute_cooperate_interaction(self, n_episode: int, episode_length: int):
        """
        return the normalized opponent sample complexity
        """
        num_inter = n_episode  # count the interactions when sampling data

        for a in self.agents:
            for i, opp_n in enumerate(a.opp_sample_num):
                opp_i = i if i < a.index else i + 1
                num_inter += (
                    opp_n
                    * self.agent_difficulties[opp_i]
                    / self.total_difficulty
                    / episode_length
                )

        return int(num_inter)

    @classmethod
    def init_from_env(
        cls,
        env,
        env_id: str,
        config: argparse.Namespace,
        alg: str = "MADDPG",
        rew_scale: float = 1.0,
        gamma: float = 0.95,
        tau: float = 0.01,
        lr: float = 0.01,
        hidden_dim: int = 64,
        model_lr: float = 0.001,
        model_hidden_dim: int = 512,
        ensemble_size: int = 7,
        env_model_buffer_size: int = 1e6,
        opp_lr: float = 0.001,
    ):

        print("Environment settings")
        print("Action", env.action_space)
        print("Observation", [obsp.shape for obsp in env.observation_space])
        print(env.agent_types)

        agent_init_params = []
        alg_types = [alg] * env.n

        n_agent = env.n

        dim_in_pol_list = []  # the input dimension of all agents
        dim_out_pol_list = []  # the ouput dimension of all agents
        dim_obs_list = []  # the observation dimentsion of all agents
        action_shape_list = []  # the action dims of all agents
        for i, (acsp, obsp, algtype) in enumerate(
            zip(env.action_space, env.observation_space, alg_types)
        ):
            dim_in_pol = obsp.shape[0]
            dim_obs_list.append(dim_in_pol)

            discrete_action = True

            def get_shape(x):
                return x.n if isinstance(x, Discrete) else sum(x.high - x.low + 1)

            # for action sample reshaping
            if isinstance(acsp, Discrete):
                action_shape = [get_shape(acsp)]
            else:  # multidiscrete
                action_shape = (acsp.high - acsp.low + 1).tolist()

            dim_out_pol = get_shape(acsp)

            dim_in_critic = dim_in_pol + dim_out_pol
            # DDPG/SAC
            agent_init_params.append(
                {
                    "dim_in_pol": dim_in_pol,
                    "dim_out_pol": dim_out_pol,
                    "dim_in_critic": dim_in_critic,
                    "agent_index": i,
                    "n_agent": n_agent,
                    "alg_type": algtype,
                    "gamma": gamma,
                    "grad_bound": config.grad_bound,
                }
            )
            if "OppMd" in algtype or "AOR" in alg_types[i]:
                agent_init_params[i].update({"opp_lr": opp_lr})
            dim_in_pol_list.append(dim_in_pol)
            dim_out_pol_list.append(dim_out_pol)
            action_shape_list.append(action_shape)

        for i in range(n_agent):
            agent_init_params[i].update({"action_shape_list": action_shape_list})
            if "Cond" in alg_types[i] or "AOR" in alg_types[i]:
                agent_init_params[i].update(
                    {
                        "dim_in_pol": agent_init_params[i]["dim_in_pol"]
                        + sum(dim_out_pol_list)
                        - agent_init_params[i]["dim_out_pol"]
                    }
                )
            if "MA" in alg_types[i] or "AOR" in alg_types[i]:
                dim_in_critic = sum(dim_obs_list) + sum(dim_out_pol_list)
                agent_init_params[i].update({"dim_in_critic": dim_in_critic})
            if "SAC" in alg_types[i]:
                agent_init_params[i].update(
                    {
                        "auto_target_entropy": True,
                    }
                )
            if "AOR" in alg_types[i]:
                agent_init_params[i].update(
                    {
                        "dim_obs_list": dim_obs_list,
                        "dim_actions": dim_out_pol_list,
                        "dim_rewards": n_agent,
                        "model_lr": model_lr,
                        "model_hidden_dim": model_hidden_dim,
                        "ensemble_size": ensemble_size,
                        "MB_batch_size": config.MB_batch_size,
                    }
                )
                agent_init_params[i].update(
                    {
                        "replay_buffer": ReplayBuffer(
                            env_model_buffer_size,
                            n_agent,
                            dim_obs_list,
                            dim_out_pol_list,
                        )
                    }
                )
            if "Opp" in alg_types[i] or "AOR" in alg_types[i]:
                tmp_dim_in_pol_list = dim_in_pol_list.copy()
                tmp_dim_in_pol_list.pop(i)
                tmp_dim_out_pol_list = dim_out_pol_list.copy()
                tmp_dim_out_pol_list.pop(i),
                agent_init_params[i].update(
                    {
                        "dim_in_opp_pols": tmp_dim_in_pol_list,
                        "dim_out_opp_pols": tmp_dim_out_pol_list,
                    }
                )

        init_dict = {
            "gamma": gamma,
            "tau": tau,
            "lr": lr,
            "rew_scale": rew_scale,
            "hidden_dim": hidden_dim,
            "alg_types": alg_types,
            "agent_init_params": agent_init_params,
            "discrete_action": discrete_action,
        }

        instance = cls(**init_dict)
        instance.init_dict = init_dict
        if env_id == "simple_schedule":
            instance.aggressive_opp_eval = True
        else:
            instance.aggressive_opp_eval = False
        return instance

    def evaluate_opp_model(
        self, sample: Tuple, logger: SummaryWriter = False, ep_i: int = None
    ):
        """
        evaluate opponent model with latest sample
        """
        obs, acts, _, _, _ = sample
        opp_pol_loss_list = []
        match_rate_list = []
        for agent_i, (a, ob) in enumerate(zip(self.agents, obs)):
            if isinstance(a, AgentOppMd):
                opp_pol_losses = []
                match_rates = []
                for opp_i, opp_pol in enumerate(a.opp_policies):
                    a_i = opp_i if opp_i < a.index else opp_i + 1
                    raw_opp_acts = opp_pol(obs[a_i])
                    action_shape_list = a.all_action_shape_list[a_i]
                    loss = 0
                    match_rate = 0
                    for i, dim in enumerate(action_shape_list):
                        real_opp_ind = torch.argmax(
                            acts[a_i][
                                :,
                                sum(action_shape_list[:i]) : sum(action_shape_list[:i])
                                + dim,
                            ],
                            dim=1,
                        )
                        raw_opp_act = raw_opp_acts[
                            :,
                            sum(action_shape_list[:i]) : sum(action_shape_list[:i])
                            + dim,
                        ]
                        loss += (
                            NLLLoss(
                                torch.log_softmax(raw_opp_act, dim=1),
                                real_opp_ind,
                            )
                            / a.dim_out_pol
                        )
                        match_rate += (
                            raw_opp_act.argmax(dim=1) == real_opp_ind
                        ).float().mean() * (
                            dim**2
                        )  # weighted actions
                    match_rates.append(match_rate / np.square(action_shape_list).sum())
                    opp_pol_losses.append(torch.sqrt(loss / len(action_shape_list)))
                a.match_rates = match_rates
                a.norm_eval_opp_pol_err = (torch.tensor(match_rates) + 1e-10) / (
                    torch.tensor(match_rates).max() + 1e-10
                )
                if self.aggressive_opp_eval:
                    a.norm_eval_opp_pol_err = torch.square(a.norm_eval_opp_pol_err)

                if logger:
                    logger.add_scalars(
                        "agent%i/evl_losses" % (agent_i),
                        {
                            "epsilon_phi^-i": torch.tensor(opp_pol_losses).sum(),
                            "match_rate_phi^-i": torch.tensor(match_rates).mean(),
                        },
                        ep_i,
                    )
                    logger.add_scalars(
                        "agent%i/evl_losses" % (agent_i),
                        dict(
                            zip(
                                [
                                    "match_rate_phi^-i,%i" % a_i
                                    for a_i in range(self.n_agent - 1)
                                ],
                                match_rates,
                            )
                        ),
                        ep_i,
                    )

                opp_pol_loss_list.append(torch.tensor(opp_pol_losses).sum())
                match_rate_list.append(torch.tensor(match_rates).mean())
        return opp_pol_loss_list, match_rate_list

    def evaluate_dynamics_model(
        self, sample: Tuple, logger: SummaryWriter = None, logger_iter: int = 0
    ) -> List[float]:
        """
        dynamics model with true environment under opponent models and current policy
        return:
            List of eval loss for each agent
        """
        obses, acts, rews, next_obses, _ = sample
        model_in = torch.cat([*obses, *acts], dim=1)
        model_target = torch.cat([*next_obses, *[r.unsqueeze(1) for r in rews]], dim=1)
        eval_loss_list = []
        for agent_i in range(self.n_agent):
            a = self.agents[agent_i]
            model_pred = a.predict(model_in)
            loss = torch.abs(model_pred - model_target).mean()
            if logger:
                logger.add_scalars(
                    "agent%i/evl_losses" % (agent_i),
                    {"abs_m_error": loss},
                    logger_iter,
                )
            eval_loss_list.append(loss)
        return eval_loss_list
