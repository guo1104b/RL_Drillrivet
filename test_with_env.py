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
    """
    在同一个 VecEnv 上顺序评估 n_episodes 幕。
    兼容 (obs, reward, done, info) 与 (obs, reward, terminated, truncated, info) 两种返回。
    """
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
    parser.add_argument("--model", type=str, default="./test2.zip",
                        help="SAC 模型文件（best_model.zip 或 final_sac_model.zip）")
    parser.add_argument("--vecnorm", type=str, default="./test2.pkl",
                        help="训练时保存的 VecNormalize 统计量文件")
    parser.add_argument("--port", type=int, default=23000,
                        help="评估连接的 CoppeliaSim 远程端口")
    parser.add_argument("--episodes", type=int, default=10,
                        help="评估的回合数")
    parser.add_argument("--deterministic", action="store_true",
                        help="是否用确定性策略（不加噪声）")
    args = parser.parse_args()

    # 1) 构建评估环境（包 Monitor，便于统计）
    eval_env = DummyVecEnv([make_eval_env(args.port)])

    # 2) 如果训练时用了 VecNormalize，这里也创建一个并加载统计量
    if os.path.exists(args.vecnorm):
        print(f"Loading vecnorm: {args.vecnorm}")
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_reward=10.0)
        eval_env.training = False  # eval 模式：不再更新统计
        eval_env.norm_reward = False
        eval_env = VecNormalize.load(args.vecnorm, eval_env)

        # 注意：VecNormalize.load 需要传入已包裹的 env 以承载统计
        # 并且要确保 training=False，否则会在评估时继续更新均值方差

    # 3) 加载模型
    print(f"Loading model: {args.model}")
    model = SAC.load(args.model, device="auto")

    # 4) 评估
    mean_return = run_eval_episodes(
        eval_env, model,
        n_episodes=args.episodes,
        deterministic=args.deterministic
    )
    print(f"[Eval] episodes={args.episodes}  mean_return={mean_return:.3f}")

    # 可选：如果需要看环境里 info 的自定义曲线（距离/角度），
    # 请确保训练阶段你已在 env.step() 的 info 里填了这些键，
    # 并在 TensorBoard 回调里记录。这里做离线评估只打印总体指标。


if __name__ == "__main__":
    main()

    # # 1) 还原 env 并加载对应步数的 VecNormalize 统计量
    # eval_env = DummyVecEnv([make_env(23001)])
    # eval_env = VecNormalize.load("./ckpt/by_step/vecnorm_step100000.pkl", eval_env)
    # eval_env.training = False
    # eval_env.norm_reward = False
    #
    # # 2) 加载模型
    # model = SAC.load("./ckpt/by_step/ckpt_step100000.zip", device="auto")
    #
    # # 3) 直接评估
    # mean_r, sr = run_eval_episodes(eval_env, model, n_episodes=5, deterministic=True)
    # print(mean_r, sr)
