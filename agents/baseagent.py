import math
from abc import abstractmethod
from typing import List, Tuple

import numpy as np
import torch
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
from utils.buffer import ReplayBuffer
from utils.networks import PtModel


MODEL_GRAD_BOUND = 200


class BaseAgent(object):
    """
    General class for agents (exploration noise)
    """

    def __init__(
        self,
        dim_out_pol: int,
        agent_index: int,
        n_agent: int,
        alg_type: str,
        rew_scale: float,
        gamma: float,
        discrete_action: bool = True,
        action_shape_list: List = [],
        dim_in_pol: int = None,
        hidden_dim: int = None,
        lr: float = None,
        grad_bound: float = 1.0,
    ):
        self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action
        self.all_action_shape_list = action_shape_list
        self.action_dim = action_shape_list[agent_index]
        self.alg_type = alg_type
        self.n_agent = n_agent
        self.index = agent_index
        self.gamma = gamma
        self.rew_scale = rew_scale
        self.dim_out_pol = dim_out_pol
        self.dim_in_pol = dim_in_pol
        self.grad_bound = grad_bound
        self.lr = lr
        self.hidden_dim = hidden_dim

        self.dev = "cpu"

        self.opp_sample_num = np.zeros((self.n_agent - 1))

    def scale_noise(self, scale: float):
        self.exploration = scale

    def random_step(self, obs: List):
        n = obs.shape[0]
        if len(self.action_dim) == 1:
            return torch.eye(self.dim_out_pol)[
                torch.randint(0, self.dim_out_pol, size=(n,))
            ].to(self.dev)
        else:
            acts = []
            for act_dim in self.action_dim:
                acts.append(torch.eye(act_dim)[torch.randint(0, act_dim, size=(n,))])
            return torch.cat(acts, dim=1).to(self.dev)

    @abstractmethod
    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        # raise NotImplementedError
        return self.random_step(obs)

    @abstractmethod
    def update(self, sample, agent_i, parallel=False, logger=None):
        """
        update the parameters
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        Outputs:
            losses for the logger
        """
        raise NotImplementedError

    @abstractmethod
    def update_targets(self, tau=None):
        raise NotImplementedError

    @abstractmethod
    def prep_training(self, device="cuda"):
        self.dev = device

    @abstractmethod
    def prep_rollouts(self, device="cpu"):
        self.dev = device

    @abstractmethod
    def get_params(self):
        # raise NotImplementedError
        return None

    @abstractmethod
    def load_params(self, params):
        # raise NotImplementedError
        pass


class AgentOppMd(BaseAgent):
    """
    Base class for agents that model the opponents
    """

    @abstractmethod
    def update_opponent(self, sample, parallel=False, logger=None):
        raise NotImplementedError

    @abstractmethod
    def get_opp_action(self, opp, obs, requires_grad=False, return_log_prob=False):
        raise NotImplementedError


class AgentCond(AgentOppMd):
    """
    Base class for agents that make decisions depending on the actions of other agents at the same turn(must have opponent models)
    """

    @abstractmethod
    def step(self, obses: List, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obses (PyTorch Variable): Observations for all agents
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        raise NotImplementedError


class AgentMB(BaseAgent):
    """
    An agent that models the environment dynamics
    (observations, actions) -> (observations, rewards)
    Have "model_rollouts" to produce rollouts under the dynamics model
    """

    def __init__(
        self,
        dim_in_pol: int,
        dim_out_pol: int,
        dim_in_critic: int,
        dim_obs_list: List[int],
        dim_actions: List[int],
        dim_rewards: int,
        agent_index: int,
        n_agent: int,
        alg_type: str,
        rew_scale: float,
        gamma: float,
        hidden_dim: int,
        lr: float,
        discrete_action: bool,
        replay_buffer: ReplayBuffer,
        action_shape_list: list = [],
        model_lr: float = 0.001,
        model_hidden_dim: int = 32,
        ensemble_size: int = 7,
        MB_batch_size: int = 4096,
        grad_bound: int = 1,
    ):
        """
        dim_obs_list (list): the agent's dimensions
        dim_actions (list): the agent's actions
        dim_rewards (int): sum of the agent's rewards

        """

        super().__init__(
            dim_out_pol,
            agent_index,
            n_agent,
            alg_type,
            rew_scale,
            gamma,
            discrete_action,
            action_shape_list,
            dim_in_pol,
            hidden_dim,
            lr,
            grad_bound,
        )

        self.dynamics_model = PtModel(
            ensemble_size,
            sum(dim_obs_list) + sum(dim_actions),
            (sum(dim_obs_list) + dim_rewards) * 2,
            model_hidden_dim,
        )
        self.model_optimizer = torch.optim.Adam(
            self.dynamics_model.parameters(), lr=model_lr
        )

        self.ensemble_size = ensemble_size
        self.dim_out_pol = dim_out_pol
        self.dim_obs_list = dim_obs_list
        self.dim_actions = dim_actions
        self.dim_rewards = dim_rewards

        self.MB_batch_size = MB_batch_size

        self.replay_buffer = replay_buffer

        self.model_device = "cpu"

        self.init_attr = {
            "action_shape_list": self.all_action_shape_list,
            "alg_type": self.alg_type,
            "n_agent": self.n_agent,
            "index": self.index,
            "dim_out_pol": self.dim_out_pol,
            "ensemble_size": self.ensemble_size,
        }  # the subclass's init_attr dict should be updated from it

    def predict(
        self,
        model_in: torch.Tensor,
        reparameterize: bool = False,
        return_normal: bool = False,
    ):
        """
        Output:
            next_state (list): predicted observations to all agents
        """
        means, stds = self.dynamics_model(model_in)
        means = means.detach()
        stds = stds.detach()
        ind = np.random.randint(0, self.ensemble_size)
        mean, std = means[ind], stds[ind]
        normal = Normal(mean, std)

        if return_normal:
            return normal

        if reparameterize:
            return normal.rsample()
        else:
            return normal.sample()

    def rollout_model(
        self,
        K: int,
        init_states: List,
        agents: List[BaseAgent],
        logger: SummaryWriter = None,
        n_iter: int = None,
    ):
        """
        return samples from the dynamics model
        Input:
            init_states (list): a list of each agent's observation list, each agent has M states from which the rollouts start, torch.tensor
            rollout_lengths (list): rollout_length for each opponent model
            agents (list): all the true agents
        Output:
            sample (list): actions should be "per agent", while others are "per state"
        process should be M-steps -> M-steps
        """

        def convert(list_tensor: List[torch.Tensor]):
            return [ts.cpu().detach().numpy() for ts in list_tensor]

        if self.norm_eval_opp_pol_err is not []:
            rollout_lengths = torch.floor(K * self.norm_eval_opp_pol_err)
        else:
            rollout_lengths = torch.zeros(len(self.opp_policies))

        if logger is not None:
            logger.add_scalars(
                "agent%i/hyper" % (self.index),
                {"min_rollout_len": torch.min(rollout_lengths.detach())},
                n_iter,
            )

        # opponent sample complexity
        for i in range(self.n_agent):
            if not i == self.index:
                if i < self.index:
                    opp_i = i
                elif i > self.index:
                    opp_i = i - 1
                self.opp_sample_num[opp_i] += (K - rollout_lengths[opp_i]) * len(
                    init_states
                )

        obs = init_states  # agent, M, feature
        state = torch.cat(obs, dim=-1)

        for k in range(K):
            acts = []  # [num_agents, M]
            for opp_i in range(self.n_agent - 1):
                agent_i = opp_i if opp_i < self.index else opp_i + 1
                if k + 1 < rollout_lengths[opp_i]:
                    acts.append(
                        self.get_opp_action(
                            self.opp_policies[opp_i], obs[agent_i], agent_i
                        )
                    )
                else:
                    # Communication
                    acts.append(
                        agents[agent_i].step(
                            obs
                            if isinstance(agents[agent_i], AgentCond)
                            else obs[agent_i],
                            explore=True,
                        )
                    )
            acts.insert(
                self.index,
                self.step(obs, explore=True)
                if isinstance(self, AgentCond)
                else self.step(obs[self.index], explore=True),
            )
            action = torch.cat(acts, dim=-1)
            model_in = torch.cat([state, action], dim=-1)
            pred = self.predict(model_in)
            next_obs = pred[..., : sum(self.dim_obs_list)]

            state = next_obs
            next_obs = [
                next_obs[
                    ...,
                    sum(self.dim_obs_list[:i]) : sum(self.dim_obs_list[: i + 1]),
                ]
                for i in range(self.n_agent)
            ]

            rews = pred[
                ...,
                sum(self.dim_obs_list) : sum(self.dim_obs_list) + self.dim_rewards,
            ]
            rews = [rews[..., i] for i in range(self.n_agent)]
            self.replay_buffer.push_agent_first(
                convert(obs),
                convert(acts),
                convert(rews),
                convert(next_obs),
                np.zeros((self.n_agent, action.shape[0])),
                action.shape[0],
            )  # need to discuss the *done*
            obs = next_obs

    def update_model(self, sample: Tuple[List], epochs: int = 1):
        """
            (observations, actions)[:dim_obs_list] <- (next_observations)
            (observations, actions)[dim_obs_list:] <- (rews)
        Outputs:
            m_loss
        """
        obs, acs, rews, next_obs, _ = sample

        state = torch.cat(obs, dim=1)
        actions = torch.cat(acs, dim=1)
        target_rews = torch.cat([r.unsqueeze(1) for r in rews], dim=1)
        target_nobs = torch.cat(next_obs, dim=1)
        model_in = torch.cat([state, actions], dim=1)
        model_target = torch.cat([target_nobs, target_rews], dim=1)

        # update normalize variables
        self.dynamics_model.fit_input_stats(model_in)

        # bootstrapping sampling
        inds = torch.randint(
            0, high=model_in.shape[0], size=(self.ensemble_size, model_in.shape[0])
        )

        n_batch = math.ceil(model_in.shape[0] / self.MB_batch_size)

        total_loss = 0

        for _ in range(epochs):
            for batch_i in range(n_batch):
                batch_inds = inds[
                    :,
                    batch_i
                    * self.MB_batch_size : min(
                        (batch_i + 1) * self.MB_batch_size, model_in.shape[0]
                    ),
                ]

                loss = 0.01 * (
                    self.dynamics_model.max_logvar.sum()
                    - self.dynamics_model.min_logvar.sum()
                )
                loss += self.dynamics_model.compute_decays()
                batch_in = model_in[batch_inds]
                batch_target = model_target[batch_inds]

                means, log_stds = self.dynamics_model(batch_in, ret_logvar=True)

                inv_stds = torch.exp(-log_stds)  # reciprocal

                train_losses = ((means - batch_target) ** 2) * inv_stds + log_stds
                train_losses = (
                    train_losses.mean(-1).mean(-1).sum()
                )  # mean operation on the ensemble_size dim makes no sense

                loss += train_losses

                self.model_optimizer.zero_grad()
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(
                    self.dynamics_model.parameters(), MODEL_GRAD_BOUND
                )
                self.model_optimizer.step()

                total_loss += loss

        return {"m_loss": total_loss / epochs / n_batch}

    def prep_training(self, device="cuda"):
        self.dynamics_model.train()
        if device == "cuda":

            def fn(x):
                return x.cuda()

        else:

            def fn(x):
                return x.cpu()

        if not self.model_device == device:
            self.dynamics_model = fn(self.dynamics_model)
            self.model_device = device

    def prep_rollouts(self, device="cpu"):
        self.dynamics_model.eval()
        if device == "cuda":

            def fn(x):
                return x.cuda()

        else:

            def fn(x):
                return x.cpu()

        if not self.model_device == device:
            self.dynamics_model = fn(self.dynamics_model)
            self.model_device = device

    def get_params(self, return_dynamics_model: bool = False):
        ret = {
            "init_attr": self.init_attr,
        }
        if return_dynamics_model:
            ret.update(
                {"dynamics_model": self.dynamics_model.state_dict(),"model_optimizer": self.model_optimizer.state_dict()}
            )
        return ret

    def load_params(self, params, load_dynamics_model:bool = False):
        if load_dynamics_model:
            self.dynamics_model.load_state_dict(params["dynamics_model"])
            self.model_optimizer.load_state_dict(params["model_optimizer"])
        for key, val in params["init_attr"].items():
            if hasattr(self, key):
                setattr(self, key, val)
