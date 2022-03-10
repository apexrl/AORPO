from typing import List
import numpy as np
import torch
from torch import Tensor


class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """

    def __init__(self, max_steps: int, num_agents: int, obs_dims: List, ac_dims: List):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((max_steps, odim)))
            self.ac_buffs.append(np.zeros((max_steps, adim)))
            self.rew_buffs.append(np.zeros(max_steps))
            self.next_obs_buffs.append(np.zeros((max_steps, odim)))
            self.done_buffs.append(np.zeros(max_steps))

        self.filled_i = (
            0  # index of first empty location in buffer (last index when full)
        )
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def clear(self):
        self.filled_i = 0
        self.curr_i = 0

    def push_sample_first(
        self, observations, actions, rewards, next_observations, dones
    ):
        nentries = observations.shape[0]  # handle multiple parallel environments
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i  # num of indices to roll over
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(
                    self.obs_buffs[agent_i], rollover, axis=0
                )
                self.ac_buffs[agent_i] = np.roll(
                    self.ac_buffs[agent_i], rollover, axis=0
                )
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i], rollover)
                self.next_obs_buffs[agent_i] = np.roll(
                    self.next_obs_buffs[agent_i], rollover, axis=0
                )
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i], rollover)
            self.curr_i = 0
            self.filled_i = self.max_steps
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][self.curr_i : self.curr_i + nentries] = np.vstack(
                observations[:, agent_i]
            )
            self.ac_buffs[agent_i][self.curr_i : self.curr_i + nentries] = np.vstack(
                actions[:, agent_i]
            )
            self.rew_buffs[agent_i][self.curr_i : self.curr_i + nentries] = rewards[
                :, agent_i
            ]
            self.next_obs_buffs[agent_i][
                self.curr_i : self.curr_i + nentries
            ] = np.vstack(next_observations[:, agent_i])
            self.done_buffs[agent_i][self.curr_i : self.curr_i + nentries] = dones[
                :, agent_i
            ]
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def push_agent_first(
        self, observations, actions, rewards, next_observations, dones, nentries
    ):
        # all inputs are in "per agent" arrangement
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i  # num of indices to roll over
            for agent_i in range(self.num_agents):
                self.obs_buffs[agent_i] = np.roll(
                    self.obs_buffs[agent_i], rollover, axis=0
                )
                self.ac_buffs[agent_i] = np.roll(
                    self.ac_buffs[agent_i], rollover, axis=0
                )
                self.rew_buffs[agent_i] = np.roll(self.rew_buffs[agent_i], rollover)
                self.next_obs_buffs[agent_i] = np.roll(
                    self.next_obs_buffs[agent_i], rollover, axis=0
                )
                self.done_buffs[agent_i] = np.roll(self.done_buffs[agent_i], rollover)
            self.curr_i = 0
            self.filled_i = self.max_steps
        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][
                self.curr_i : self.curr_i + nentries
            ] = observations[agent_i]
            self.ac_buffs[agent_i][self.curr_i : self.curr_i + nentries] = actions[
                agent_i
            ]
            self.rew_buffs[agent_i][self.curr_i : self.curr_i + nentries] = rewards[
                agent_i
            ]
            self.next_obs_buffs[agent_i][
                self.curr_i : self.curr_i + nentries
            ] = next_observations[agent_i]
            self.done_buffs[agent_i][self.curr_i : self.curr_i + nentries] = dones[
                agent_i
            ]
        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0

    def sample_from_inds(self, inds, to_gpu=False, norm_rews=True):
        if to_gpu:
            cast = lambda x: torch.tensor(
                x, requires_grad=False, dtype=torch.float
            ).cuda()
        else:
            cast = lambda x: torch.tensor(x, requires_grad=False, dtype=torch.float)
        if norm_rews:
            ret_rews = [
                cast(
                    (
                        self.rew_buffs[i][inds]
                        - self.rew_buffs[i][: self.filled_i].mean()
                    )
                    / self.rew_buffs[i][: self.filled_i].std()
                )
                for i in range(self.num_agents)
            ]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        return (
            [cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
            [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
            ret_rews,
            [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
            [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)],
        )

    def latest_sample(self, N, to_gpu=False, norm_rews=True):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        np.random.shuffle(inds)
        return self.sample_from_inds(inds, to_gpu, norm_rews)

    def episode_sample(self, N, to_gpu=False, norm_rews=True):
        all_inds = np.arange(0, self.filled_i, 50)
        inds = np.random.choice(all_inds, N // 50, replace=False)
        np.random.shuffle(inds)
        inds = np.concatenate([np.arange(i, i + 50) for i in inds], axis=0)
        return self.sample_from_inds(inds, to_gpu, norm_rews)

    def sample(self, N, to_gpu=False, norm_rews=True, replace: bool = False):
        inds = np.random.choice(np.arange(self.filled_i), size=N, replace=replace)
        return self.sample_from_inds(inds, to_gpu, norm_rews)

    def get_average_rewards(self, ep_length, n_roll):
        if self.filled_i == self.max_steps:
            inds = np.arange(
                self.curr_i - ep_length * n_roll, self.curr_i
            )  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - ep_length * n_roll), self.curr_i)
        return [self.rew_buffs[i][inds].sum() / n_roll for i in range(self.num_agents)]
