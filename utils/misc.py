import os
from typing import Callable, List

# from torch.autograd import Variable
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
import random
from torch import nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter


def t_2_n(tensor: torch.Tensor) -> np.ndarray:
    return tensor.cpu().detach().numpy()


def n_2_t(
    array: np.ndarray,
    requires_grad: bool = False,
    copy=True,
    dtype: torch.dtype = torch.float,
    device: torch.device = "cpu",
):
    if copy:
        return torch.tensor(
            array, requires_grad=requires_grad, dtype=dtype, device=device
        )
    else:
        assert requires_grad is False, "copy from numpy"
        assert torch.device(device).type == "cpu", "copy from numpy"
        return torch.from_numpy(array)


def soft_update(target, source, tau):
    # https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11
    """
    Perform DDPG soft update (move target params toward source based on weight
    factor tau)
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
        tau (float, 0 < x < 1): Weight factor for update
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    # https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15
    """
    Copy network parameters from source to target
    Inputs:
        target (torch.nn.Module): Net to copy parameters to
        source (torch.nn.Module): Net whose parameters to copy
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def onehot_from_logits(logits, eps=0.0, return_log_prob=False):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    if eps == 0.0:
        if return_log_prob:
            inds = torch.argmax(logits, dim=1, keepdim=True)
            logits = F.softmax(logits, dim=1)
            log_prob = torch.log(logits.gather(1, inds) + 1e-10)
            return argmax_acs, log_prob
        return argmax_acs
    # get random actions in one-hot form
    rand_acs = torch.eye(logits.shape[1], requires_grad=False)[
        np.random.choice(range(logits.shape[1]), size=logits.shape[0])
    ]
    # chooses between best and random actions using epsilon greedy

    if return_log_prob:
        log_prob = torch.log(logits / torch.sum(logits) + 1e-10)
        return (
            torch.stack(
                [
                    argmax_acs[i] if r > eps else rand_acs[i]
                    for i, r in enumerate(torch.rand(logits.shape[0]))
                ]
            ),
            log_prob,
        )
    else:
        return torch.stack(
            [
                argmax_acs[i] if r > eps else rand_acs[i]
                for i, r in enumerate(torch.rand(logits.shape[0]))
            ]
        )


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor, device="cpu"):
    # modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    """Sample from Gumbel(0, 1)"""
    U = tens_type(*shape).clone().detach().requires_grad_(False).uniform_().to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    # modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    """Draw a sample from the Gumbel-Softmax distribution"""
    # print(logits)
    y = logits + sample_gumbel(
        logits.shape, tens_type=type(logits.data), device=logits.device
    )
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0, hard=False, return_log_prob=False):
    # modified for PyTorch from https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    softmax_output = y
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y  # for the correct gradients
    if return_log_prob:
        inds = torch.argmax(y, dim=1, keepdim=True)
        log_prob = torch.log(softmax_output.gather(1, inds) + 1e-10)
        return y, log_prob
    return y


def swish(x):
    # from https://github.com/quanvuong/handful-of-trials-pytorch/blob/master/config/utils.py
    return x * torch.sigmoid(x)


def truncated_normal(size, std):
    # from https://github.com/quanvuong/handful-of-trials-pytorch/blob/master/config/utils.py
    # We use TF to implement initialization function for neural network weight because:
    # 1. Pytorch doesn't support truncated normal
    # 2. This specific type of initialization is important for rapid progress early in training in cartpole

    # Do not allow tf to use gpu memory unnecessarily
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True

    sess = tf.Session(config=cfg)
    val = sess.run(tf.random.truncated_normal(shape=size, stddev=std))

    # Close the session and free resources
    sess.close()

    return torch.tensor(val, dtype=torch.float32)


def get_affine_params(ensemble_size, in_features, out_features):
    # from https://github.com/quanvuong/handful-of-trials-pytorch/blob/master/config/utils.py
    w = truncated_normal(
        size=(ensemble_size, in_features, out_features),
        std=1.0 / (2.0 * np.sqrt(in_features)),
    )
    w = nn.Parameter(w)

    b = nn.Parameter(torch.zeros(ensemble_size, 1, out_features, dtype=torch.float32))

    return w, b


def apply_with_grad(tensor, fn):
    """
    move the grad data and the tensor data to device with fn in-place
    """
    tensor.data = fn(tensor.data)
    if tensor.grad is not None:
        tensor.grad.data = fn(tensor.grad.data)
    return tensor


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_multi_discrete_action(
    action: torch.Tensor,
    action_shape_list: List[np.ndarray],
    fn: Callable,
    return_log_prob: bool = False,
):
    if len(action_shape_list) == 1:
        return fn(action, return_log_prob=return_log_prob)
    else:
        actions = torch.tensor([], device=action.device)
        if return_log_prob:
            log_prob = 1
            for i, dim in enumerate(action_shape_list):
                act, log_prob_ = fn(
                    action[
                        :, sum(action_shape_list[:i]) : sum(action_shape_list[:i]) + dim
                    ],
                    return_log_prob=True,
                )
                actions = torch.cat([actions, act], dim=1)
                log_prob *= log_prob_
            return actions, log_prob
        else:
            for i, dim in enumerate(action_shape_list):
                actions = torch.cat(
                    [
                        actions,
                        fn(
                            action[
                                :,
                                sum(action_shape_list[:i]) : sum(action_shape_list[:i])
                                + dim,
                            ],
                            return_log_prob=False,
                        ),
                    ],
                    dim=1,
                )
            return actions


class optimizer_lr_multistep_scheduler:
    def __init__(
        self,
        steps: List[int],
        optimizers: List[torch.optim.Optimizer],
        gamma: float = 0.5,
        logger: SummaryWriter = None,
    ) -> None:
        self.step_i = 0
        self.gamma = gamma
        self.steps = steps
        self.optimizers = optimizers
        self.logger = logger

    def step(self, episode: int):
        if self.step_i < len(self.steps) and episode >= self.steps[self.step_i]:
            for optimizer in self.optimizers:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= self.gamma
            self.step_i += 1
            lr = self.optimizers[0].param_groups[0]["lr"]
            if self.logger:
                self.logger.add_scalars("hypers/", {"model_lr": lr}, episode)
            print(f"decrease dynamics lr to {lr}")
