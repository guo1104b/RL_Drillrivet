# test trained model
# eval.py
import os
import argparse
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gymnasium.wrappers import TimeLimit

from enverr2 import DrillRivetEnv


def run_eval_episodes(vec_env, model, n_episodes=3, deterministic=True):
   
    ep_returns, ep_success = [], []

    for ep in range(n_episodes):
        obs = vec_env.reset()
        terminated = False
        truncated = False
        done = False
        ep_ret, steps = 0.0, 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            res = vec_env.step(action)
            # obs, reward, terminated, truncated, infos = res
            obs, reward, done, infos = res
            # print(f"done: {done}")

            ep_ret += float(reward)
            steps += 1

        ep_returns.append(ep_ret)

    return float(np.mean(ep_returns))


def make_env(port):
    # return lambda: Monitor(DrillRivetEnv(port))
    return lambda: Monitor(TimeLimit(DrillRivetEnv(port), max_episode_steps=250))

def make_eval_env(port):
    env = DrillRivetEnv(port)
    env = TimeLimit(env, max_episode_steps=250)
    return lambda: Monitor(env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="./final.zip",
                        help="SAC best_model.zip or final_sac_model.zip）")
    parser.add_argument("--vecnorm", type=str, default="./final.pkl",
                        help="VecNormalize file")
    parser.add_argument("--port", type=int, default=23000,
                        help="CoppeliaSim remote port")
    parser.add_argument("--episodes", type=int, default=10,
                        help="number of episodes for test")
    parser.add_argument("--deterministic", action="store_true",
                        help="wheter or not use deterministic strategy")
    args = parser.parse_args()

    eval_env = DummyVecEnv([make_eval_env(args.port)])
    
    if os.path.exists(args.vecnorm):
        print(f"Loading vecnorm: {args.vecnorm}")
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_reward=10.0)
        eval_env.training = False  
        eval_env.norm_reward = False
        eval_env = VecNormalize.load(args.vecnorm, eval_env)


    print(f"Loading model: {args.model}")
    model = SAC.load(args.model, device="auto")

    mean_return = run_eval_episodes(
        eval_env, model,
        n_episodes=args.episodes,
        deterministic=args.deterministic
    )
    print(f"[Eval] episodes={args.episodes}  mean_return={mean_return:.3f}")

    
if __name__ == "__main__":
    main()
