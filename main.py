import argparse
import json
import os
import pprint
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import tqdm
from gym.spaces import Box, Discrete
from torch.utils.tensorboard import SummaryWriter

from agents import *
from algorithms.maframework import MA_Controller
from utils.buffer import ReplayBuffer
from utils.make_env import make_env, make_parallel_env
from utils.misc import n_2_t, optimizer_lr_multistep_scheduler, setup_seed, t_2_n

torch.set_default_dtype(torch.float)

algorithm_list = [
    "DDPG",
    "MADDPGOppMd",
    "AORDPG",
    "SAC",
    "MASACOppMd",
    "AORPO",
]

Cooperative_env_list = [
    "simple_speaker_listener",
    "simple_spread",
    "simple_schedule",
]


def run_eval(
    env_id: str,
    controller: MA_Controller,
    episode_length: int = 25,
    n_episode: int = 10,
):
    controller.prep_rollouts("cpu")
    rets = []
    for ep_i in range(n_episode):
        env = make_env(env_id)
        env.seed(ep_i)
        obs = env.reset()
        ret = 0
        for _ in range(episode_length):
            torch_obs = [
                n_2_t(obs[a_i]).reshape(1, -1) for a_i in range(controller.n_agent)
            ]
            torch_act = controller.step(torch_obs, explore=False)
            act = [t_2_n(a_act).reshape(-1) for a_act in torch_act]
            n_obs, rew, _, _ = env.step(act)
            obs = n_obs

            ret += np.mean(rew)
        rets.append(ret)
        env.close()

    return np.mean(rets)


def run(config: argparse.Namespace) -> Dict:
    model_dir = (
        Path(config.model_dir) / config.algorithm / config.env_id / config.model_name
    )
    if not model_dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in model_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / "logs"
    os.makedirs(log_dir)
    print("run dir", run_dir)
    with open(os.path.join(run_dir, "config"), "w") as cf:
        json.dump(config.__dict__, cf)
    logger = SummaryWriter(str(log_dir))

    setup_seed(config.seed)

    eval_env = make_env(config.env_id)
    n_agent = eval_env.n

    torch.set_num_threads(config.n_training_thread)

    env = make_parallel_env(
        config.env_id, config.n_sample_thread, config.seed, config.discrete_action
    )
    controller = MA_Controller.init_from_env(
        env,
        config.env_id,
        config,
        alg=config.algorithm,
        rew_scale=config.rew_scale,
        gamma=config.gamma,
        tau=config.tau,
        lr=config.lr,
        hidden_dim=config.hidden_dim,
        model_lr=config.model_lr,
        model_hidden_dim=config.model_hidden_dim,
        ensemble_size=config.ensemble_size,
        env_model_buffer_size=int(config.env_model_buffer_size),
        opp_lr=config.opp_lr,
    )
    replay_buffer = ReplayBuffer(
        int(config.buffer_size),
        n_agent,
        [obsp.shape[0] for obsp in env.observation_space],
        [
            acsp.shape[0]
            if isinstance(acsp, Box)
            else (
                acsp.n if isinstance(acsp, Discrete) else sum(acsp.high - acsp.low + 1)
            )
            for acsp in env.action_space
        ],
    )

    if any([isinstance(a, AgentMB) for a in controller.agents]):
        models_lr_scheduler = optimizer_lr_multistep_scheduler(
            config.model_lr_schedule_steps,
            [a.model_optimizer for a in controller.agents],
            logger=logger,
        )

    n_step = 0
    K = 1
    env_rate = 1
    model_trained = False
    eval_ret = 0

    eval_ret_dict = {}

    # train the policy and opponent models but not the dynamics model
    warmup_episode = int(1e6)
    if any([isinstance(a, AgentMB) for a in controller.agents]):
        warmup_episode = config.n_model_warmup_episode
        print("initialize episodes", warmup_episode)

    for epoch in range(0, config.n_epoch):
        print(
            f"tag {config.model_name}, seed {config.seed}, {config.algorithm}, epoch {epoch+1}, {config.episode_per_epoch} episodes, {config.n_sample_thread} threads"
        )

        # the noise is changed each epoch, but is the noise is not used at all
        # eplr_remain_frac = max(0, config.n_eplr_epoch - epoch) / config.n_eplr_epoch
        # noise_scale = (
        #     config.noise_scale_final
        #     + (config.noise_scale_start - config.noise_scale_final) * eplr_remain_frac
        # )
        # controller.scale_noise(noise_scale)
        # logger.add_scalars(
        #     "hypers/", {"noise_scale": noise_scale}, logger_dynamics_iter
        # )

        episode_2_epoch_iter = tqdm.trange(
            0,
            config.episode_per_epoch,
            config.n_sample_thread,
        )
        for episode_in_epoch in episode_2_epoch_iter:
            episode_i = epoch * config.episode_per_epoch + (
                episode_in_epoch + config.n_sample_thread
            )
            logger_dynamics_iter = episode_i
            logger_oppo_iter = controller.compute_cooperate_interaction(
                episode_i, config.episode_length
            )
            obs = env.reset()
            # thread, agent, feature
            controller.prep_rollouts(device="cpu")
            for _ in range(config.episode_length):
                # rearrange observations to be per agent, and convert to torch Variable
                torch_obs = [
                    n_2_t(np.vstack(obs[:, a_i]), requires_grad=False)
                    for a_i in range(n_agent)
                ]
                # [agent, tensor: thread, feature]

                # get actions as torch Variables
                torch_action = controller.step(torch_obs, explore=True)
                # agent, thread, action_dim

                # convert actions to numpy arrays
                agent_first_action = [
                    t_2_n(agent_action) for agent_action in torch_action
                ]
                # rearrange actions to be per environment
                thread_first_action = [
                    [agent_action[i] for agent_action in agent_first_action]
                    for i in range(config.n_sample_thread)
                ]
                # thread, agent, action_dim
                next_obs, rewards, dones, infos = env.step(thread_first_action)
                thread_first_action = np.stack(thread_first_action, axis=0)

                replay_buffer.push_sample_first(
                    obs, thread_first_action, rewards, next_obs, dones
                )

                obs = next_obs
                n_step += config.n_sample_thread

                if (
                    len(replay_buffer) > config.batch_size * 5
                    and n_step % config.dynamics_model_update_step_interval
                    < config.n_sample_thread
                ):
                    # update model
                    if any([isinstance(a, AgentMB) for a in controller.agents]):
                        if config.cuda:
                            controller.prep_training(device="cuda")
                        else:
                            controller.prep_training(device="cpu")

                        sample = replay_buffer.sample(
                            len(replay_buffer),
                            to_gpu=config.cuda,
                        )
                        i = 0
                        min_i = i
                        min_evl_loss = [float("inf") for _ in range(n_agent)]
                        best_models = [
                            controller.get_dynamics_model(a_i) for a_i in range(n_agent)
                        ]
                        sample_train = [
                            [a_obj[a_obj.shape[0] // 5 :] for a_obj in obj]
                            for obj in sample
                        ]
                        sample_eval = [
                            [a_obj[: a_obj.shape[0] // 5].cpu() for a_obj in obj]
                            for obj in sample
                        ]
                        while True:
                            if config.cuda:
                                controller.prep_training(device="cuda")
                            else:
                                controller.prep_training(device="cpu")
                            controller.update_dynamics_models(
                                sample_train,
                                logger_iter=logger_dynamics_iter,
                                logger=logger,
                                epochs=1,
                            )

                            controller.prep_rollouts(device="cpu")
                            evl_loss = controller.evaluate_dynamics_model(sample_eval)

                            i += 1
                            for a_i in range(n_agent):
                                if min_evl_loss[a_i] > evl_loss[a_i]:
                                    min_evl_loss[a_i] = evl_loss[a_i]
                                    min_i = i
                                    best_models[a_i] = controller.get_dynamics_model(
                                        a_i
                                    )
                            if min_i <= i - 5 or i > 50:
                                break
                        if config.DEBUG:
                            print(f"min_i in dynamics model update {min_i}")

                        for a_i in range(n_agent):
                            controller.set_dynamics_model(a_i, best_models[a_i])
                        controller.evaluate_dynamics_model(
                            sample_eval, logger, logger_dynamics_iter
                        )
                        controller.prep_rollouts(device="cpu")

                        models_lr_scheduler.step(epoch * config.episode_per_epoch)
                        model_trained = True

                if (
                    len(replay_buffer) >= config.batch_size * 5
                    and (n_step % config.update_step_interval) < config.n_sample_thread
                ):
                    if any(
                        [isinstance(a, AgentOppMd) for a in controller.agents]
                    ) or any([isinstance(a, AgentMB) for a in controller.agents]):
                        batch_size = min(config.batch_size * 3, len(replay_buffer))
                        latest_sample = replay_buffer.latest_sample(
                            batch_size, to_gpu=config.cuda
                        )
                        latest_sample_train = [
                            [a_obj[a_obj.shape[0] // 5 :] for a_obj in obj]
                            for obj in latest_sample
                        ]
                        latest_sample_eval = [
                            [a_obj[: a_obj.shape[0] // 5].cpu() for a_obj in obj]
                            for obj in latest_sample
                        ]
                    # update opponents
                    if any([isinstance(a, AgentOppMd) for a in controller.agents]):
                        i = 0
                        max_i = 0
                        max_match_rates = [-float("inf") for _ in range(n_agent)]
                        best_opponents = [
                            controller.get_opponent_model(a_i) for a_i in range(n_agent)
                        ]
                        while True:
                            if config.cuda:
                                controller.prep_training(device="cuda")
                            else:
                                controller.prep_training(device="cpu")

                            controller.update_opponent_models(
                                latest_sample_train,
                                logger=logger,
                                epochs=1,
                                logger_iter=logger_dynamics_iter,
                            )

                            controller.prep_rollouts(device="cpu")

                            (
                                _,
                                match_rate_list,
                            ) = controller.evaluate_opp_model(latest_sample_eval)
                            # matching rate for each agent
                            i += 1

                            for a_i in range(n_agent):
                                if max_match_rates[a_i] < match_rate_list[a_i]:
                                    max_match_rates[a_i] = match_rate_list[a_i]
                                    max_i = i
                                    best_opponents[a_i] = controller.get_opponent_model(
                                        a_i
                                    )
                            if max_i <= i - 5 or i > 50:
                                break
                        if config.DEBUG:
                            print(f"max_i in opponent model update {max_i}")
                        for a_i in range(n_agent):
                            controller.set_opponent_model(a_i, best_opponents[a_i])
                        controller.evaluate_opp_model(
                            latest_sample_eval,
                            logger=logger,
                            ep_i=n_step // config.episode_length,
                        )
                        controller.prep_rollouts(device="cpu")

                    # use model
                    if (
                        episode_i >= warmup_episode
                        and any([isinstance(a, AgentMB) for a in controller.agents])
                        and model_trained
                    ):
                        controller.prep_rollouts(device="cpu")

                        # model rollout
                        K = int(
                            min(
                                config.K,
                                max(
                                    K,
                                    K
                                    + (epoch - config.A)
                                    / ((config.B - config.A))
                                    * (config.K - K),
                                ),
                            )
                        )
                        if K > 0:
                            if config.gpu_rollout_model:
                                # gpu model rollout
                                controller.prep_rollouts(device="cuda")
                            else:
                                # cpu model rollout
                                controller.prep_rollouts(device="cpu")

                            init_states = replay_buffer.sample(
                                config.M, to_gpu=config.gpu_rollout_model, replace=True
                            )[0]
                            controller.rollout_models(
                                K, init_states, logger, n_step // config.episode_length
                            )
                        controller.prep_rollouts(device="cpu")

                    # update policy
                    if config.cuda:
                        controller.prep_training(device="cuda")
                    else:
                        controller.prep_training(device="cpu")
                    for a_i in range(controller.n_agent):
                        if (
                            episode_i >= warmup_episode
                            and (isinstance(controller.agents[a_i], AgentMB))
                            and K > 0
                            and model_trained
                        ):
                            env_rate = config.Env_rate_start + min(
                                1.0, epoch / (config.Env_rate_n_epoch)
                            ) * (config.Env_rate_finish - config.Env_rate_start)

                            env_batch_size = int(config.batch_size * env_rate)
                            model_batch_size = config.batch_size - env_batch_size
                            # TODO: add warning!
                            if (
                                config.env_model_buffer_size // config.batch_size
                                < config.G
                            ):
                                G = config.G
                            else:
                                G = min(
                                    len(controller.agents[a_i].replay_buffer)
                                    // (config.batch_size),
                                    config.G,
                                )
                            for _ in range(G):
                                env_sample = replay_buffer.sample(
                                    env_batch_size, to_gpu=config.cuda
                                )
                                model_sample = controller.agents[
                                    a_i
                                ].replay_buffer.sample(
                                    model_batch_size, to_gpu=config.cuda
                                )
                                sample = env_sample
                                # concat for each component
                                for s, m_s in zip(sample, model_sample):
                                    for i, (a_s, a_m_s) in enumerate(zip(s, m_s)):
                                        s[i] = torch.cat([a_s, a_m_s], dim=0)
                                controller.update(
                                    sample,
                                    a_i,
                                    logger=logger,
                                    logger_iter=logger_dynamics_iter,
                                )
                        else:
                            sample = replay_buffer.sample(
                                config.batch_size, to_gpu=config.cuda
                            )
                            controller.update(
                                sample,
                                a_i,
                                logger=logger,
                                logger_iter=logger_dynamics_iter,
                            )

                    controller.update_all_targets()

                    controller.prep_rollouts(device="cpu")

            ep_rews = replay_buffer.get_average_rewards(
                config.episode_length, config.n_sample_thread
            )
            for a_i, a_ep_rew in enumerate(ep_rews):
                logger.add_scalars(
                    "agent%i/mean_episode_return" % a_i,
                    {"dynamics_interaction": a_ep_rew},
                    logger_dynamics_iter,
                )
                logger.add_scalars(
                    "agent%i/mean_episode_return" % a_i,
                    {"opponent_interaction": a_ep_rew},
                    logger_oppo_iter,
                )

            episode_2_epoch_iter.set_description(
                f"train return {np.mean(ep_rews):.4f}  eval ret {eval_ret:.4f}"
            )

            logger.add_scalars(
                "hypers/",
                {"K": K, "Env_rate": env_rate},
                logger_dynamics_iter,
            )
        n_episode = (epoch + 1) * config.episode_per_epoch
        if (n_episode % config.save_episode_interval) == 0:
            os.makedirs(run_dir / "incremental", exist_ok=True)
            controller.save(
                run_dir
                / "incremental"
                / ("model_dynamics%i.pt" % (logger_dynamics_iter))
            )
            controller.save(
                run_dir / "incremental" / ("model_opponent%i.pt" % (logger_oppo_iter))
            )

        eval_ret = run_eval(config.env_id, controller)
        episode_2_epoch_iter.set_description(
            f"train ret {np.mean(ep_rews):.4f} eval ret {eval_ret:.4f}"
        )

        eval_ret_dict[(epoch + 1) * config.episode_per_epoch] = eval_ret
        logger.add_scalars(
            "eval",
            {"mean_episode_return_dynamics_interaction": eval_ret},
            (epoch + 1) * config.episode_per_epoch,
        )
        logger.add_scalars(
            "eval",
            {"mean_episode_return_opponent_interaction": eval_ret},
            controller.compute_cooperate_interaction(
                (epoch + 1) * config.episode_per_epoch, config.episode_length
            ),
        )

    eval_env.close()
    env.close()
    logger.close()

    os.makedirs(run_dir / "incremental", exist_ok=True)
    controller.save(run_dir / "incremental" / ("model.pt"))

    return {curr_run: eval_ret_dict}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument(
        "env_id", help="Name of environment", type=str, choices=Cooperative_env_list
    )
    parser.add_argument("model_name", help="Name of directory to store")
    parser.add_argument("--seed", default=1, type=int, help="Random seed")
    parser.add_argument("--episode_length", default=25, type=int)

    # agent
    parser.add_argument(
        "--algorithm", default="AORPO", type=str, choices=algorithm_list
    )
    parser.add_argument("--discrete_action", default=True, type=bool)

    # model-free
    parser.add_argument("--n_training_thread", default=10, type=int)
    parser.add_argument("--n_sample_thread", default=15, type=int)
    parser.add_argument("--n_epoch", default=150, help="Epochs", type=int)
    parser.add_argument("--episode_per_epoch", default=300, type=int)
    # eposide = n_epoch * episode_per_epoch
    parser.add_argument(
        "--update_step_interval", default=100, help="policy update interval", type=int
    )

    parser.add_argument("--buffer_size", default=int(1e6), type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--grad_bound", default=1.0, type=float)  # not used
    parser.add_argument(
        "--batch_size", default=1024, type=int, help="Batch size for model training"
    )

    parser.add_argument("--rew_scale", default=5.0, type=float)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.95, type=float)
    parser.add_argument("--cuda", action="store_true")

    parser.add_argument("--noise_scale_start", default=0.3, type=float)
    parser.add_argument("--noise_scale_final", default=0.0, type=float)
    parser.add_argument(
        "--n_eplr_epoch", help="exploration epochs number", default=100, type=int
    )

    # model-based
    # opponent model
    parser.add_argument("--opp_lr", default=0.0005, type=float)
    # dynamics model
    # model-learning
    parser.add_argument("--model_hidden_dim", default=256, type=int)
    parser.add_argument("--ensemble_size", default=8, type=int)
    parser.add_argument(
        "--MB_batch_size", default=4096, type=int, help="Batch size for model training"
    )
    parser.add_argument("--model_lr", default=0.001, type=float)
    parser.add_argument(
        "--model_lr_schedule_steps",
        nargs="+",
        type=int,
        default=[],
        help="steps for model lr decreasing",
    )
    parser.add_argument(
        "--dynamics_model_update_step_interval",
        default=75 * 25,
        help="dynamics model update interval",
        type=int,
    )
    # model-usage
    parser.add_argument("--K", default=6, type=int)
    parser.add_argument("--M", default=2048, type=int)
    parser.add_argument("--env_model_buffer_lastest", action="store_true")
    parser.add_argument("--gpu_rollout_model", action="store_true")
    parser.add_argument(
        "--G",
        default=20,
        type=int,
        help="policy update G times in model-based dyna-style methods",
    )

    parser.add_argument("--n_model_warmup_episode", default=1500, type=int)
    parser.add_argument("--Env_rate_n_epoch", default=40, type=int)
    parser.add_argument("--Env_rate_start", default=0.5, type=float)
    parser.add_argument("--Env_rate_finish", default=0.5, type=float)
    parser.add_argument("--A", default=15, type=int)
    parser.add_argument("--B", default=100, type=int)

    # log
    parser.add_argument("--load_path", type=str)
    parser.add_argument("--model_dir", default="./models")
    parser.add_argument("--save_episode_interval", default=300, type=int)

    parser.add_argument("--DEBUG", action="store_true")

    config = parser.parse_args()

    config.device = "cuda" if config.cuda and torch.cuda.is_available() else "cpu"

    if config.env_model_buffer_lastest:
        config.env_model_buffer_size = config.M
    else:
        config.env_model_buffer_size = (
            config.M
            * config.dynamics_model_update_step_interval
            // config.update_step_interval
        )

    # sanity check

    # assert (
    #     config.n_eplr_epoch <= config.n_epoch
    # ), "exploration epoch steps <= total epoch steps"

    assert (
        config.episode_per_epoch % config.n_sample_thread == 0
    ), "multi-thread sampling"

    if config.model_lr_schedule_steps is None:
        config.model_lr_schedule_steps = []
    print("=" * 20 + "config" + "=" * 20)
    pprint.pprint(config.__dict__)

    eval_dicts = {}

    # for seed in range(5):
    #     config.seed = seed
    #     eval_dict = run(config)
    #     eval_dicts.update(eval_dict)
    eval_dict = run(config)
    eval_dicts.update(eval_dict)

    result_name = "-".join([run_id.replace("run", "") for run_id in eval_dicts.keys()])
    result_path = f"./results/{config.model_name}/{config.env_id}/{config.algorithm}/run{result_name}"
    os.makedirs(result_path, exist_ok=True)

    with open(os.path.join(result_path, "results.json"), "w") as f:
        json.dump(eval_dicts, f)
        print(f"eval results saved in {result_path}")
