from abc import abstractmethod
from async_timeout import enum

import numpy as np
import torch
from torch import tensor
from torch.optim import Adam

from utils.misc import (
    gumbel_softmax,
    hard_update,
    onehot_from_logits,
    soft_update,
    get_multi_discrete_action,
)
from utils.networks import MLPNetwork

from .baseagent import *

MSELoss = torch.nn.MSELoss()
NLLLoss = torch.nn.NLLLoss()

OPPMD_GRAD_BOUND = 1


class DDPGAgent(BaseAgent):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """

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
        action_shape_list: list = [],
        grad_bound: float = 1.0,
    ):
        """
        Inputs:
            dim_in_pol (int): number of dimensions for policy input
            dim_out_pol (int): number of dimensions for policy output
            dim_in_critic (int): number of dimensions for critic input
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
            action_shape_list,
            grad_bound=grad_bound,
        )
        self.dim_in_pol = dim_in_pol

        self.policy = MLPNetwork(
            dim_in_pol,
            dim_out_pol,
            hidden_dim=hidden_dim,
        )
        self.critic = MLPNetwork(dim_in_critic, 1, hidden_dim=hidden_dim)
        self.target_policy = MLPNetwork(
            dim_in_pol,
            dim_out_pol,
            hidden_dim=hidden_dim,
        )
        self.target_critic = MLPNetwork(dim_in_critic, 1, hidden_dim=hidden_dim)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)

        self.pol_dev = "cpu"  # device for policies
        self.critic_dev = "cpu"  # device for critics
        self.trgt_pol_dev = "cpu"  # device for target policies
        self.trgt_critic_dev = "cpu"  # device for target critics

        self.init_attr = {
            "index": self.index,
            "action_shape_list": self.all_action_shape_list,
            "alg_type": self.alg_type,
            "n_agent": self.n_agent,
        }

    def step(self, obs: torch.Tensor, explore: bool = False, return_raw: bool = False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        if return_raw:
            return action
        if explore:

            def processfun(x, return_log_prob):
                return gumbel_softmax(x, hard=True)

        else:
            processfun = onehot_from_logits

        action = get_multi_discrete_action(
            action, self.all_action_shape_list[self.index], processfun
        )
        return action

    def get_target_action(self, obs: torch.Tensor):
        action = self.target_policy(obs)
        action = get_multi_discrete_action(
            action, self.all_action_shape_list[self.index], onehot_from_logits
        )
        return action

    def update(self, sample: Tuple[List]):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agents (list of agents): instances include *self*
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        Outputs:
            pol_loss
            cf_loss
        """
        obs, acs, rews, next_obs, dones = sample

        # update critic
        if "MA" in self.alg_type or "AOR" in self.alg_type:
            all_trgt_acs = []
            for a_i, nobs in enumerate(next_obs):
                if a_i == self.index:
                    all_trgt_acs.append(self.get_target_action(nobs).detach())
                else:
                    opp_i = a_i if a_i < self.index else a_i - 1
                    all_trgt_acs.append(
                        DDPGAgentOppMd.get_opp_action(
                            self, self.opp_policies[opp_i], nobs, a_i
                        ).detach()
                    )

            trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)
        else:  # DDPG
            trgt_vf_in = torch.cat(
                (next_obs[self.index], self.get_target_action(next_obs[self.index])),
                dim=1,
            )

        target_value = rews[self.index].view(-1, 1) + self.gamma * self.target_critic(
            trgt_vf_in
        )

        if "MA" in self.alg_type or "AOR" in self.alg_type:
            vf_in = torch.cat((*obs, *acs), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[self.index], acs[self.index]), dim=1)

        actual_value = self.critic(vf_in)
        vf_loss = MSELoss(actual_value, target_value.detach())
        self.critic_optimizer.zero_grad()
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_bound)
        self.critic_optimizer.step()

        # update policy
        curr_pol_out = self.policy(obs[self.index])
        curr_pol_vf_in = get_multi_discrete_action(
            curr_pol_out,
            self.all_action_shape_list[self.index],
            lambda x, return_log_prob: gumbel_softmax(
                x, hard=True, return_log_prob=return_log_prob
            ),
        )
        if "MA" in self.alg_type or "AOR" in self.alg_type:
            all_pol_acs = []
            for a_i, ob in enumerate(obs):
                if a_i == self.index:
                    all_pol_acs.append(curr_pol_vf_in)
                else:
                    opp_i = a_i if a_i < self.index else a_i - 1
                    all_pol_acs.append(
                        DDPGAgentOppMd.get_opp_action(
                            self, self.opp_policies[opp_i], ob, a_i
                        ).detach()
                    )
            vf_in = torch.cat((*obs, *all_pol_acs), dim=1)
        else:  # DDPG
            vf_in = torch.cat((obs[self.index], curr_pol_vf_in), dim=1)
        pol_loss = -self.critic(vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3  # regularizer
        self.policy_optimizer.zero_grad()
        pol_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_bound)
        self.policy_optimizer.step()
        return {
            "cf_loss": vf_loss,
            "pol_loss": pol_loss,
        }

    def update_targets(self, tau: float):
        soft_update(self.target_critic, self.critic, tau)
        soft_update(self.target_policy, self.policy, tau)

    def prep_training(self, device="cuda"):
        self.policy.train()
        self.critic.train()
        self.target_policy.train()
        self.target_critic.train()
        if device == "cuda":

            def fn(x):
                return x.cuda()

        else:

            def fn(x):
                return x.cpu()

        if not self.pol_dev == device:
            self.policy = fn(self.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            self.target_policy = fn(self.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device="cpu"):
        self.policy.eval()

        if device == "cuda":

            def fn(x):
                return x.cuda()

        else:

            def fn(x):
                return x.cpu()

        if not self.pol_dev == device:
            self.policy = fn(self.policy)
            self.pol_dev = device

    def get_params(self):
        return {
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "target_policy": self.target_policy.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "init_attr": self.init_attr,
        }

    def load_params(self, params):
        self.policy.load_state_dict(params["policy"])
        self.critic.load_state_dict(params["critic"])
        self.target_policy.load_state_dict(params["target_policy"])
        self.target_critic.load_state_dict(params["target_critic"])
        self.policy_optimizer.load_state_dict(params["policy_optimizer"])
        self.critic_optimizer.load_state_dict(params["critic_optimizer"])
        for key, val in params["init_attr"].items():
            if hasattr(self, key):
                setattr(self, key, val)


class DDPGAgentOppMd(DDPGAgent, AgentOppMd):
    """
    General class for DDPG agents that model the opponent policies (policy, critic, target policy, target
    critic, opponent policies, exploration noise)
    The opponent models are also updated in the MADDPG class.
    """

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
        action_shape_list: list = [],
        grad_bound: float = 1.0,
        opp_lr: float = 0.001,
    ):
        """
        Inputs:
            dim_in_pol (int): number of dimensions for policy input
            dim_out_pol (int): number of dimensions for policy output
            dim_in_critic (int): number of dimensions for critic input
            agent_index (int): the index of the agent
            dim_in_opp_pols (list of int): numbers of dimensions for opponents' policy input
            dim_out_opp_pols (list of int): numbers of dimensions for opponents' policy output
        """
        super(DDPGAgentOppMd, self).__init__(
            dim_in_pol,
            dim_out_pol,
            dim_in_critic,
            agent_index,
            n_agent,
            alg_type,
            rew_scale,
            gamma,
            hidden_dim=hidden_dim,
            lr=lr,
            discrete_action=discrete_action,
            action_shape_list=action_shape_list,
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

        self.opp_pol_dev = "cpu"

    def get_opp_action(
        self,
        opp: torch.nn.Module,
        obs: torch.Tensor,
        agent_ind: int,
        requires_grad: bool = False,
        return_log_prob: bool = False,
    ):
        """
        Inputs:
            requires_grad (bool):
                for discrete action, true means gumbel_softmax with *hard=True*, otherwise means onehot_from_logits.
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
        obs, acs, rews, next_obs, dones = sample
        # update opponent models
        opp_pol_losses = []

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
            for i, dim in enumerate(action_shape_list):
                action_ = action[
                    :, sum(action_shape_list[:i]) : sum(action_shape_list[:i]) + dim
                ]
                opp_pl_out = torch.log_softmax(action_, dim=1)
                actual_opp_pl = acs[agent_i][
                    :, sum(action_shape_list[:i]) : sum(action_shape_list[:i]) + dim
                ]
                acutal_opp_pl_ind = torch.argmax(actual_opp_pl, dim=1)  # shape: [R, N]
                opp_pol_loss += NLLLoss(opp_pl_out, acutal_opp_pl_ind).mean()
            opp_pol_loss += opp_log_pi.mean() * 0.1
            opp_pl_optimizer.zero_grad()
            opp_pol_loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(opp_pl.parameters(), OPPMD_GRAD_BOUND)
            opp_pl_optimizer.step()
            opp_pol_losses.append(opp_pol_loss)
        return {"opp_pol_loss": torch.mean(torch.tensor(opp_pol_losses))}

    def prep_training(self, device="cuda"):
        super().prep_training(device)
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
        super().prep_rollouts(device)
        for (
            opp_pol
        ) in self.opp_policies:  # opponent policies are also need in the rollouts
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
        ret = super().get_params()
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
        super().load_params(params)
        for i in range(len(self.opp_policies)):
            self.opp_policies[i].load_state_dict(params["opp_policies"][i])
            self.opp_policy_optimizers[i].load_state_dict(
                params["opp_policy_optimizers"][i]
            )


class DDPGAgentOppMdCond(DDPGAgentOppMd, AgentCond):
    """
    General class for DDPG agents that model the opponent policies (policy, critic, target policy, target
    critic, opponent policies, exploration noise)
    The opponent models are also updated in the MADDPG class.
    """

    def step(self, obs: torch.Tensor, explore: bool = False, return_raw: bool = False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obses (PyTorch Variable): Observations for all agents
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        agent_obs = obs[self.index]
        obs_ = obs.copy()
        obs_.pop(self.index)
        opp_acts = [
            self.get_opp_action(opp_pl, ob, i if i < self.index else i + 1)
            for i, (opp_pl, ob) in enumerate(zip(self.opp_policies, obs_))
        ]

        pol_in = torch.cat([agent_obs, *opp_acts], dim=1)
        action = self.policy(pol_in)
        if return_raw:
            return action
        if explore:

            def processfun(x, return_log_prob):
                return gumbel_softmax(x, hard=True)

        else:
            processfun = onehot_from_logits
        action = get_multi_discrete_action(
            action, self.all_action_shape_list[self.index], processfun
        )
        return action

    def get_target_action(self, obs: torch.Tensor):
        """
        obs (list): observations of all agents
        """
        agent_obs = obs[self.index]
        obs_ = obs.copy()
        obs_.pop(self.index)
        opp_acts = [
            self.get_opp_action(opp_pl, ob, i if i < self.index else i + 1)
            for i, (opp_pl, ob) in enumerate(zip(self.opp_policies, obs_))
        ]
        pol_in = torch.cat([agent_obs, *opp_acts], dim=1)
        action = self.target_policy(pol_in)

        action = get_multi_discrete_action(
            action, self.all_action_shape_list[self.index], onehot_from_logits
        )
        return action

    def update(self, sample: Tuple[List]):
        """
        Update parameters of agent model based on sample from replay buffer
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
            cf_loss
        """
        obs, acs, rews, next_obs, dones = sample

        # update critic
        policies = self.opp_policies.copy()
        policies.insert(self.index, self.policy)
        all_trgt_acs = []
        for i, (a, nobs) in enumerate(zip(policies, next_obs)):
            if i == self.index:
                all_trgt_acs.append(self.get_target_action(next_obs))
            else:
                all_trgt_acs.append(self.get_opp_action(a, nobs, i))

        trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1)

        target_value = rews[self.index].view(-1, 1) + self.gamma * self.target_critic(
            trgt_vf_in
        )

        vf_in = torch.cat((*obs, *acs), dim=1)

        actual_value = self.critic(vf_in)
        vf_loss = MSELoss(actual_value, target_value.detach())
        self.critic_optimizer.zero_grad()
        vf_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_bound)
        self.critic_optimizer.step()

        # update policy
        agent_obs = obs[self.index]
        obs_ = obs.copy()
        obs_.pop(self.index)
        opp_acts = [
            self.get_opp_action(opp_pl, ob, i if i < self.index else i + 1)
            for i, (opp_pl, ob) in enumerate(zip(self.opp_policies, obs_))
        ]
        pol_in = torch.cat([agent_obs, *opp_acts], dim=1)
        curr_pol_out = self.policy(pol_in)
        curr_pol_vf_in = get_multi_discrete_action(
            curr_pol_out,
            self.all_action_shape_list[self.index],
            lambda x, return_log_prob: gumbel_softmax(
                x, hard=True, return_log_prob=return_log_prob
            ),
        )
        all_pol_acs = []
        for i, (a, ob) in enumerate(zip(policies, obs)):
            if i == self.index:
                all_pol_acs.append(curr_pol_vf_in)
            else:
                all_pol_acs.append(self.get_opp_action(a, ob, i))
        vf_in = torch.cat((*obs, *all_pol_acs), dim=1)

        pol_loss = -self.critic(vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3  # regularizer
        self.policy_optimizer.zero_grad()
        pol_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_bound)
        self.policy_optimizer.step()
        return {
            "cf_loss": vf_loss,
            "pol_loss": pol_loss,
        }

    def prep_rollouts(self, device="cpu"):
        super().prep_rollouts(device)
        for (
            opp_pol
        ) in self.opp_policies:  # opponent policies are also need in the rollouts
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


class DDPGAgentOppMdCondMB(DDPGAgentOppMdCond, AgentMB):
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
        replay_buffer,
        model_lr=0.001,
        model_hidden_dim=32,
        ensemble_size=7,
        action_shape_list=[],
        opp_lr=0.001,
        grad_bound: float = 1.0,
        MB_batch_size: int = 4096,
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

        DDPGAgentOppMdCond.__init__(
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
            action_shape_list=action_shape_list,
            opp_lr=opp_lr,
        )

        init_attr.update(self.init_attr)
        self.init_attr = init_attr

    def get_params(self):
        ret = DDPGAgentOppMdCond.get_params(self)
        ret.update(AgentMB.get_params(self))
        return ret

    def load_params(self, params):
        DDPGAgentOppMdCond.load_params(self, params)
        AgentMB.load_params(self, params)

    def prep_rollouts(self, device="cpu"):
        DDPGAgentOppMdCond.prep_rollouts(self, device)
        AgentMB.prep_rollouts(self, device)

    def prep_training(self, device="cuda"):
        DDPGAgentOppMdCond.prep_training(self, device)
        AgentMB.prep_training(self, device)
