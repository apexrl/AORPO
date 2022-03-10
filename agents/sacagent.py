from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from utils.misc import (
    apply_with_grad,
    get_multi_discrete_action,
    gumbel_softmax,
    hard_update,
    onehot_from_logits,
    soft_update,
)
from utils.networks import MLPNetwork

from .baseagent import *

MSELoss = nn.MSELoss()
NLLLoss = nn.NLLLoss()

POLICY_GRAD_BOUND = 1
CRITIC_GRAD_BOUND = 40
OPPMD_GRAD_BOUND = 1


class SACAgent(BaseAgent):
    def __init__(
        self,
        dim_in_pol: int,
        dim_out_pol: int,
        dim_in_critic: int,
        agent_index: int,
        n_agent: int,
        alg_type: str,
        rew_scale: float,
        gamma: float,
        hidden_dim: int,
        lr: float,
        discrete_action: bool,
        auto_target_entropy: bool = True,
        target_entropy: int = None,
        reparameterize: bool = True,
        action_shape_list: list = [],
        grad_bound: float = 1.0,
    ):
        """
        Inputs:
            dim_in_pol (int): number of dimensions for policy input
            dim_out_pol (int): number of dimensions for policy output
            dim_in_critic (int): number of dimensions for critic input
            auto_target_entropy(bool): whether to use auto target entropy
            target_entropy(float): the target entropy used
        """
        BaseAgent.__init__(
            self,
            dim_out_pol,
            agent_index,
            n_agent,
            alg_type,
            rew_scale,
            gamma,
            discrete_action,
            action_shape_list=action_shape_list,
            grad_bound=grad_bound,
        )
        self.dim_in_pol = dim_in_pol
        self.policy = MLPNetwork(
            dim_in_pol,
            dim_out_pol,
            hidden_dim=hidden_dim,
        )
        self.critic1 = MLPNetwork(dim_in_critic, 1, hidden_dim=hidden_dim)
        self.critic2 = MLPNetwork(dim_in_critic, 1, hidden_dim=hidden_dim)
        self.target_critic1 = MLPNetwork(dim_in_critic, 1, hidden_dim=hidden_dim)
        self.target_critic2 = MLPNetwork(dim_in_critic, 1, hidden_dim=hidden_dim)
        hard_update(self.target_critic1, self.critic1)
        hard_update(self.target_critic2, self.critic2)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=lr)
        self.auto_target_entropy = auto_target_entropy
        self.reparameterize = reparameterize
        self.target_entropy = (
            -np.prod(dim_out_pol).item() if target_entropy is None else target_entropy
        )

        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = Adam([self.log_alpha], lr=lr)

        self.policy_dev = "cpu"  # device for policies
        self.critic_dev = "cpu"  # device for critics
        self.trgt_critic_dev = "cpu"  # device for target critics
        self.trgt_value_dev = "cpu"  # device for target value

        self.init_attr = {
            "action_shape_list": self.all_action_shape_list,
            "alg_type": self.alg_type,
            "n_agent": self.n_agent,
            "auto_target_entropy": self.auto_target_entropy,
            "reparameterize": self.reparameterize,
            "target_entropy": self.target_entropy,
            "log_alpha": self.log_alpha,
            "index": self.index,
        }

    def step(
        self,
        obs: torch.Tensor,
        explore: bool = False,
        return_log_prob: bool = False,
        return_raw: bool = False,
    ):
        if explore:

            def processfun(x, return_log_prob):
                return gumbel_softmax(x, hard=True, return_log_prob=return_log_prob)

        else:
            processfun = onehot_from_logits

        if self.discrete_action:
            action = self.policy(obs)
            if return_raw:
                return action
            if return_log_prob:
                action, log_prob = get_multi_discrete_action(
                    action,
                    self.all_action_shape_list[self.index],
                    processfun,
                    return_log_prob=True,
                )
            else:
                action = get_multi_discrete_action(
                    action,
                    self.all_action_shape_list[self.index],
                    processfun,
                    return_log_prob=False,
                )
        if return_log_prob:
            return action, log_prob
        else:
            return action

    def update(self, sample: Tuple[List]):
        """
        update the parameters
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        Outputs:
            pol_loss
            vf_loss
            cf1_loss
            cf2_loss
        """
        obs, acs, rews, next_obs, dones = sample

        # update policy
        raw_action = self.policy(obs[self.index])
        action, log_pi = get_multi_discrete_action(
            raw_action,
            self.all_action_shape_list[self.index],
            lambda x, return_log_prob: gumbel_softmax(
                x, hard=True, return_log_prob=return_log_prob
            ),
            return_log_prob=True,
        )

        if "MA" in self.alg_type or "AOR" in self.alg_type:
            all_pol_acs = []
            for a_i, n_ob in enumerate(obs):
                if a_i == self.index:
                    all_pol_acs.append(action)
                else:
                    opp_i = a_i if a_i < self.index else a_i - 1
                    all_pol_acs.append(
                        SACAgentOppMd.get_opp_action(
                            self, self.opp_policies[opp_i], n_ob, a_i
                        ).detach()
                    )
            # Mutli-agent Q (s, \vec{a})
            critic_in = torch.cat([*obs, *all_pol_acs], dim=1)
        else:  # SAC
            critic_in = torch.cat([obs[self.index], action], dim=1)

        critic1_output = self.critic1(critic_in)
        critic2_output = self.critic2(critic_in)
        min_critic_output = torch.min(critic1_output, critic2_output)

        # update alpha
        if self.auto_target_entropy:
            alpha_loss = -(
                self.log_alpha * ((log_pi + self.target_entropy).detach())
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp().detach()
        else:
            alpha = 1

        # update policy
        if self.reparameterize:
            policy_kl = (alpha * log_pi - min_critic_output).mean()
        else:  # it seems useless
            raise NotImplementedError
        policy_loss = policy_kl + (raw_action**2).mean() * 1e-3  # regularizer

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), POLICY_GRAD_BOUND
        )
        self.policy_optimizer.step()

        # update critic
        if "MA" in self.alg_type or "AOR" in self.alg_type:
            critic_in = torch.cat([*obs, *acs], dim=1)
        else:
            critic_in = torch.cat([obs[self.index], acs[self.index]], dim=1)

        critic1_output = self.critic1(critic_in)
        critic2_output = self.critic2(critic_in)

        next_obs_action, next_obs_log_pi = self.step(
            next_obs[self.index], explore=False, return_log_prob=True
        )
        if "MA" in self.alg_type or "AOR" in self.alg_type:
            next_obs_actions = []
            for a_i, n_ob in enumerate(next_obs):
                if a_i == self.index:
                    next_obs_actions.append(next_obs_action.detach())
                else:
                    opp_i = a_i if a_i < self.index else a_i - 1
                    next_obs_actions.append(
                        self.get_opp_action(
                            self.opp_policies[opp_i], n_ob, a_i
                        ).detach()
                    )
            target_critic_in = torch.cat([*next_obs, *next_obs_actions], dim=1)
        else:
            target_critic_in = torch.cat(
                [next_obs[self.index], next_obs_action.detach()], dim=1
            )
        min_target_critic_output = (
            torch.min(
                self.target_critic1(target_critic_in),
                self.target_critic2(target_critic_in),
            )
            - alpha * next_obs_log_pi.detach()
        )

        target = (
            self.rew_scale * rews[self.index].view(-1, 1)
            + self.gamma * min_target_critic_output
        )

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()

        critic1_loss = MSELoss(critic1_output, target.detach())
        critic2_loss = MSELoss(critic2_output, target.detach())

        critic1_loss.backward()
        critic2_loss.backward()
        grad1 = nn.utils.clip_grad_norm_(self.critic1.parameters(), CRITIC_GRAD_BOUND)
        grad2 = nn.utils.clip_grad_norm_(self.critic2.parameters(), CRITIC_GRAD_BOUND)
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        return {
            "alpha": alpha,
            "log_pol_pi": (alpha * log_pi).mean(),
            "pol_critic": min_critic_output.mean(),
            "pol_loss": policy_loss,
            "cf1_loss": critic1_loss,
            "cf2_loss": critic2_loss,
        }

    def update_targets(self, tau: float):
        soft_update(self.target_critic1, self.critic1, tau)
        soft_update(self.target_critic2, self.critic2, tau)

    def prep_training(self, device="cuda"):
        self.policy.train()

        self.critic1.train()
        self.critic2.train()

        self.target_critic1.train()
        self.target_critic2.train()
        if device == "cuda":

            def fn(x):
                return x.cuda()

        else:

            def fn(x):
                return x.cpu()

        if not self.policy_dev == device:
            self.policy = fn(self.policy)
            self.policy_dev = device

        if not self.critic_dev == device:
            self.critic1 = fn(self.critic1)
            self.critic2 = fn(self.critic2)
            self.critic_dev = device

        if not self.trgt_critic_dev == device:
            self.target_critic1 = fn(self.target_critic1)
            self.target_critic2 = fn(self.target_critic2)
            self.trgt_critic_dev = device

        self.log_alpha = apply_with_grad(self.log_alpha, fn)

    def prep_rollouts(self, device="cpu"):
        self.policy.eval()

        if device == "cuda":

            def fn(x):
                return x.cuda()

        else:

            def fn(x):
                return x.cpu()

        if not self.policy_dev == device:
            self.policy = fn(self.policy)
            self.policy_dev = device

        self.log_alpha = apply_with_grad(self.log_alpha, fn)

    def get_params(self):
        return {
            "policy": self.policy.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "init_attr": self.init_attr,
        }

    def load_params(self, params):
        self.policy.load_state_dict(params["policy"])
        self.critic1.load_state_dict(params["critic1"])
        self.critic2.load_state_dict(params["critic2"])
        self.target_critic1.load_state_dict(params["target_critic1"])
        self.target_critic2.load_state_dict(params["target_critic2"])
        self.policy_optimizer.load_state_dict(params["policy_optimizer"])
        self.critic1_optimizer.load_state_dict(params["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(params["critic2_optimizer"])
        self.alpha_optimizer.load_state_dict(params["alpha_optimizer"])
        for key, val in params["init_attr"].items():
            if hasattr(self, key):
                setattr(self, key, val)


class SACAgentOppMd(SACAgent, AgentOppMd):
    def __init__(
        self,
        dim_in_pol: int,
        dim_out_pol: int,
        dim_in_critic: int,
        agent_index: int,
        dim_in_opp_pols: List[int],
        dim_out_opp_pols: List[int],
        n_agent: int,
        alg_type: str,
        rew_scale: float,
        gamma: float,
        hidden_dim: int,
        lr: float,
        discrete_action: bool,
        auto_target_entropy: bool = True,
        target_entropy: int = None,
        reparameterize: bool = True,
        action_shape_list: list = [],
        grad_bound: float = 1.0,
        opp_lr: float = 0.001,
    ):
        """
        Inputs:
            dim_in_pol (int): number of dimensions for policy input
            dim_out_pol (int): number of dimensions for policy output
            dim_in_critic (int): number of dimensions for critic input
            auto_target_entropy(bool): whether to use auto target entropy
            target_entropy(float): the target entropy used
        """

        super().__init__(
            dim_in_pol,
            dim_out_pol,
            dim_in_critic,
            agent_index,
            n_agent,
            alg_type,
            rew_scale,
            gamma,
            hidden_dim,
            lr,
            discrete_action,
            auto_target_entropy,
            target_entropy,
            reparameterize,
            action_shape_list,
            grad_bound=grad_bound,
        )
        self.opp_lr = opp_lr

        self.opp_policies = []
        for num_in_opp_pol, num_out_opp_pol in zip(dim_in_opp_pols, dim_out_opp_pols):
            self.opp_policies.append(
                MLPNetwork(
                    num_in_opp_pol,
                    num_out_opp_pol,
                    hidden_dim=hidden_dim,
                )
            )
        self.opp_policy_optimizers = [
            Adam(opp_policy.parameters(), lr=self.opp_lr)
            for opp_policy in self.opp_policies
        ]

        self.norm_evl_opp_pol_err = []
        self.opp_pol_dev = "cpu"  # device for opponent policies

    def get_opp_action(
        self,
        opp: nn.Module,
        obs: Tensor,
        agent_ind: int,
        requires_grad: bool = False,
        return_log_prob: bool = False,
    ):
        """
        Inputs:
            requires_grad (bool):
            for discrete action, true means gumbel_softmax with *hard=True*, otherwise means onehot_from_logits.
            for continous action, true means *reparameterize=True*, otherwise means *reparameterize=False*
        """
        if requires_grad:

            def processfun(x, return_log_prob):
                return gumbel_softmax(x, hard=True, return_log_prob=return_log_prob)

        else:
            processfun = onehot_from_logits

        raw_action = opp(obs)
        if return_log_prob:

            action, log_prob = get_multi_discrete_action(
                raw_action,
                self.all_action_shape_list[agent_ind],
                processfun,
                return_log_prob=True,
            )
        else:
            action = get_multi_discrete_action(
                raw_action, self.all_action_shape_list[agent_ind], processfun
            )
        if return_log_prob:
            return action, log_prob
        else:
            return action

    def update_opponent(self, sample: Tuple[List]):
        """
        update the parameters of opponent models
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        Outputs:
            opp_pol_loss
        """
        obs, acs, _, _, _ = sample
        # update opponent models
        opp_pol_losses = []
        match_rates = []
        for opp_i, (opp_pl, opp_pl_optimizer) in enumerate(
            zip(self.opp_policies, self.opp_policy_optimizers)
        ):
            agent_i = opp_i if opp_i < self.index else opp_i + 1
            action = opp_pl(obs[agent_i])
            opp_pl_out, opp_log_pi = get_multi_discrete_action(
                action,
                self.all_action_shape_list[agent_i],
                lambda x, return_log_prob: gumbel_softmax(
                    x, hard=True, return_log_prob=return_log_prob
                ),
                return_log_prob=True,
            )

            opp_pol_loss = 0
            action_shape_list = self.all_action_shape_list[agent_i]
            for opp_i, dim in enumerate(action_shape_list):
                action_ = action[
                    :,
                    sum(action_shape_list[:opp_i]) : sum(action_shape_list[:opp_i])
                    + dim,
                ]
                opp_pl_out = torch.log_softmax(action_, dim=1)
                actual_opp_pl = acs[agent_i][
                    :,
                    sum(action_shape_list[:opp_i]) : sum(action_shape_list[:opp_i])
                    + dim,
                ]
                acutal_opp_pl_ind = torch.argmax(actual_opp_pl, dim=1)
                opp_pol_loss += NLLLoss(opp_pl_out, acutal_opp_pl_ind).mean()
                opp_pl_ind = torch.argmax(action_, dim=-1)
                match_rate = (opp_pl_ind == acutal_opp_pl_ind).float().mean()
            opp_pol_loss += opp_log_pi.mean() * 0.1
            opp_pl_optimizer.zero_grad()
            opp_pol_loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(opp_pl.parameters(), OPPMD_GRAD_BOUND)

            opp_pl_optimizer.step()
            opp_pol_losses.append(opp_pol_loss)
            match_rates.append(match_rate)
        return {
            "opp_pol_loss": torch.mean(torch.tensor(opp_pol_losses)),
            "opp_match_rate": torch.mean(torch.tensor(match_rates)),
        }

    def prep_training(self, device: str = "cuda"):
        SACAgent.prep_training(self, device)
        for opp_pol in self.opp_policies:
            opp_pol.train()
        if device == "cuda":

            def fn(x):
                return x.cuda()

        else:

            def fn(x):
                return x.cpu()

        if not self.opp_pol_dev == device:
            for i in range(len(self.opp_policies)):
                self.opp_policies[i] = fn(self.opp_policies[i])
            self.opp_pol_dev = device

    def prep_rollouts(self, device="cpu"):
        SACAgent.prep_rollouts(self, device)
        for opp_pol in self.opp_policies:
            opp_pol.eval()
        if device == "cuda":

            def fn(x):
                return x.cuda()

        else:

            def fn(x):
                return x.cpu()

        if not self.opp_pol_dev == device:
            for i in range(len(self.opp_policies)):
                self.opp_policies[i] = fn(self.opp_policies[i])
            self.opp_pol_dev = device

    def get_params(self):
        ret = SACAgent.get_params(self)
        ret.update(
            {
                "opp_policies": [opp_pl.state_dict() for opp_pl in self.opp_policies],
                "opp_policy_optimizers": [
                    opp_pl_optimizer.state_dict()
                    for opp_pl_optimizer in self.opp_policy_optimizers
                ],
            }
        )
        return ret

    def load_params(self, params):
        SACAgent.load_params(self, params)
        for i in range(len(self.opp_policies)):
            self.opp_policies[i].load_state_dict(params["opp_policies"][i])
            self.opp_policy_optimizers[i].load_state_dict(
                params["opp_policy_optimizers"][i]
            )


class SACAgentOppMdCond(SACAgentOppMd, AgentCond):
    def step(
        self,
        obs: Tensor,
        explore: bool = False,
        return_log_prob: bool = False,
        return_raw: bool = False,
    ):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (list): Observations for all agents
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        agent_obs = obs[self.index]
        obs_ = obs.copy()
        obs_.pop(self.index)
        opp_acts = [
            self.get_opp_action(opp_pl, ob, i if i < self.index else i + 1).detach()
            for i, (opp_pl, ob) in enumerate(zip(self.opp_policies, obs_))
        ]
        pol_in = torch.cat([agent_obs, *opp_acts], dim=1)

        if explore:

            def processfun(x, return_log_prob):
                return gumbel_softmax(x, hard=True, return_log_prob=return_log_prob)

        else:
            processfun = onehot_from_logits
        action = self.policy(pol_in)
        if return_raw:
            return action
        if return_log_prob:
            action, log_prob = get_multi_discrete_action(
                action,
                self.all_action_shape_list[self.index],
                processfun,
                return_log_prob=True,
            )
        else:
            action = get_multi_discrete_action(
                action, self.all_action_shape_list[self.index], processfun
            )
        if return_log_prob:
            return action, log_prob
        else:
            return action

    def update(self, sample: Tuple[List]):
        """
        update the parameters
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        Outputs:
            pol_loss
            cf1_loss
            cf2_loss
        """
        obs, acs, rews, next_obs, dones = sample
        # update policy
        agent_obs = obs[self.index]
        obs_ = obs.copy()
        obs_.pop(self.index)
        opp_acts = [
            self.get_opp_action(opp_pl, ob, i if i < self.index else i + 1).detach()
            for i, (opp_pl, ob) in enumerate(zip(self.opp_policies, obs_))
        ]
        pol_in = torch.cat([agent_obs, *opp_acts], dim=1)
        raw_action = self.policy(pol_in)
        action, log_pi = get_multi_discrete_action(
            raw_action,
            self.all_action_shape_list[self.index],
            lambda x, return_log_prob: gumbel_softmax(
                x, hard=True, return_log_prob=return_log_prob
            ),
            return_log_prob=True,
        )

        opp_acts.insert(self.index, action)
        all_pol_acs = opp_acts
        critic_in = torch.cat([*obs, *all_pol_acs], dim=1)
        critic1_output = self.critic1(critic_in)
        critic2_output = self.critic2(critic_in)
        min_critic_output = torch.min(critic1_output, critic2_output)

        # update alpha
        if self.auto_target_entropy:
            alpha_loss = -(
                self.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp().detach()
        else:
            alpha = 1

        # update policy
        if self.reparameterize:
            policy_kl = (alpha * log_pi - min_critic_output).mean()
        else:  # it seems useless
            raise NotImplementedError
        policy_loss = policy_kl + (raw_action**2).mean() * 1e-3  # regularizer

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), POLICY_GRAD_BOUND
        )
        self.policy_optimizer.step()

        # update critic
        critic_in = torch.cat([*obs, *acs], dim=1)
        critic1_output = self.critic1(critic_in)
        critic2_output = self.critic2(critic_in)

        next_obs_action, next_obs_log_pi = self.step(
            next_obs, explore=False, return_log_prob=True
        )
        next_obs_actions = []
        for a_i, n_ob in enumerate(next_obs):
            if a_i == self.index:
                next_obs_actions.append(next_obs_action.detach())
            else:
                opp_i = a_i if a_i < self.index else a_i - 1
                next_obs_actions.append(
                    self.get_opp_action(self.opp_policies[opp_i], n_ob, a_i).detach()
                )
        target_critic_in = torch.cat([*next_obs, *next_obs_actions], dim=1)
        min_target_critic_output = (
            torch.min(
                self.target_critic1(target_critic_in),
                self.target_critic2(target_critic_in),
            )
            - alpha * next_obs_log_pi
        )

        target = (
            self.rew_scale * rews[self.index].view(-1, 1)
            + self.gamma * min_target_critic_output
        )

        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()

        critic1_loss = MSELoss(critic1_output, target.detach())
        critic2_loss = MSELoss(critic2_output, target.detach())

        critic1_loss.backward()
        critic2_loss.backward()
        grad1 = nn.utils.clip_grad_norm_(self.critic1.parameters(), CRITIC_GRAD_BOUND)
        grad2 = nn.utils.clip_grad_norm_(self.critic2.parameters(), CRITIC_GRAD_BOUND)
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        return {
            "alpha": alpha,
            "log_pol_pi": (alpha * log_pi).mean(),
            "pol_critic": min_critic_output.mean(),
            "pol_loss": policy_loss,
            "cf1_loss": critic1_loss,
            "cf2_loss": critic2_loss,
        }


class SACAgentOppMdCondMB(SACAgentOppMdCond, AgentMB):
    def __init__(
        self,
        dim_in_pol,
        dim_out_pol,
        dim_in_critic,
        dim_obs_list,
        dim_actions,
        dim_rewards,
        agent_index,
        dim_in_opp_pols,
        dim_out_opp_pols,
        n_agent,
        alg_type,
        rew_scale,
        gamma,
        hidden_dim,
        lr,
        discrete_action,
        auto_target_entropy=True,
        target_entropy=None,
        reparameterize=True,
        replay_buffer=None,
        model_lr=0.001,
        model_hidden_dim=32,
        ensemble_size=7,
        action_shape_list=[],
        MB_batch_size: int = 4096,
        grad_bound: float = 1.0,
        opp_lr=0.001,
    ):
        AgentMB.__init__(
            self,
            dim_in_pol,
            dim_out_pol,
            dim_in_critic,
            dim_obs_list,
            dim_actions,
            dim_rewards,
            agent_index,
            n_agent,
            alg_type,
            rew_scale,
            gamma,
            hidden_dim,
            lr,
            discrete_action,
            replay_buffer,
            model_lr=model_lr,
            model_hidden_dim=model_hidden_dim,
            ensemble_size=ensemble_size,
            action_shape_list=action_shape_list,
            grad_bound=grad_bound,
            MB_batch_size=MB_batch_size,
        )

        init_attr = self.init_attr

        SACAgentOppMdCond.__init__(
            self,
            dim_in_pol,
            dim_out_pol,
            dim_in_critic,
            agent_index,
            dim_in_opp_pols,
            dim_out_opp_pols,
            n_agent,
            alg_type,
            rew_scale,
            gamma,
            hidden_dim=hidden_dim,
            lr=lr,
            discrete_action=discrete_action,
            auto_target_entropy=auto_target_entropy,
            target_entropy=target_entropy,
            reparameterize=reparameterize,
            action_shape_list=action_shape_list,
            opp_lr=opp_lr,
        )

        init_attr.update(self.init_attr)
        self.init_attr = init_attr

    def get_params(self):
        ret = SACAgentOppMdCond.get_params(self)
        ret.update(AgentMB.get_params(self))
        return ret

    def load_params(self, params):
        SACAgentOppMdCond.load_params(self, params)
        AgentMB.load_params(self, params)

    def prep_rollouts(self, device="cpu"):
        SACAgentOppMdCond.prep_rollouts(self, device)
        AgentMB.prep_rollouts(self, device)

    def prep_training(self, device="cuda"):
        SACAgentOppMdCond.prep_training(self, device)
        AgentMB.prep_training(self, device)
