import argparse
import numpy as np
import tqdm
from algorithms.maframework import MA_Controller
from utils.make_env import make_env
from utils.misc import n_2_t, t_2_n

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
    render: bool = False,
):
    controller.prep_rollouts("cpu")
    rets = []
    for ep_i in tqdm.trange(n_episode):
        env = make_env(env_id)
        env.seed(ep_i)
        obs = env.reset()
        ret = 0
        for _ in range(episode_length):
            if render:
                env.render()
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument(
        "env_id", help="Name of environment", type=str, choices=Cooperative_env_list
    )
    parser.add_argument("load_path", type=str)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--n_episode", default=10, type=int)
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()

    controller = MA_Controller.init_from_save(args.load_path)

    eval_ret = run_eval(
        args.env_id, controller, args.episode_length, args.n_episode, args.render
    )

    print(f"Evaluation mean return: {eval_ret:.4f}")
