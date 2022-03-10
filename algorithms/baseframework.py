from typing import List, Tuple
import torch
from abc import abstractmethod
from torch.utils.tensorboard import SummaryWriter

MSELoss = torch.nn.MSELoss()


class BaseFramework(object):
    """
    Wrapper class for controller in multi-agent task
    """

    def __init__(
        self,
        alg_types: List[str],
        rew_scale: float = 1.0,
        gamma: float = 0.95,
        tau: float = 0.01,
        lr: float = 0.01,
        hidden_dim: int = 64,
        discrete_action: int = True,
    ):
        """
        Inputs:
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.n_agent = len(alg_types)
        self.alg_types = alg_types
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.rew_sacle = rew_scale
        self.hidden_dim = hidden_dim
        self.niter = 0

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    @abstractmethod
    def step(self, observations: List, explore: bool = False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        pass

    @abstractmethod
    def update(self, sample: Tuple, agent_i: int, logger: SummaryWriter = None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        pass

    @abstractmethod
    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        pass

    @abstractmethod
    def prep_training(self, device="cuda"):
        pass

    @abstractmethod
    def prep_rollouts(self, device="cuda"):
        pass

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
    def init_from_save(cls, model_path: str) -> "BaseFramework":
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(model_path, map_location=torch.device("cpu"))
        instance = cls(**save_dict["init_dict"])
        instance.init_dict = save_dict["init_dict"]
        for a, params in zip(instance.agents, save_dict["agent_params"]):
            a.load_params(params)
        return instance

    @classmethod
    def init_from_env(
        cls,
        env,
        alg: str = "AORPO",
        rew_scale: float = 1.0,
        gamma: float = 0.95,
        tau: float = 0.01,
        lr: float = 0.01,
        hidden_dim: int = 64,
    ) -> "BaseFramework":
        """
        Instantiate instance of this class from multi-agent environment
        """
        pass
