import os
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback

from enverr3 import DrillRivetEnv

def sync_envs_normalization(src: VecNormalize, dest: VecNormalize) -> None:
    """
    手动同步两个 VecNormalize 的观测归一化统计量
    （均值、方差、计数），避免 eval_env 和 train_env 不一致。

    :param src: 训练用的 VecNormalize
    :param dest: 评估用的 VecNormalize
    """
    if not isinstance(src, VecNormalize) or not isinstance(dest, VecNormalize):
        raise TypeError("Both envs must be VecNormalize instances")

    dest.obs_rms.mean = src.obs_rms.mean.copy()
    dest.obs_rms.var = src.obs_rms.var.copy()
    dest.obs_rms.count = src.obs_rms.count

class CrossProcEvalCallback(EvalCallback):
    def _on_step(self) -> bool:
        # 每次评估前把统计从训练 env 拷到 eval env
        due = self.eval_freq > 0 and (self.n_calls % self.eval_freq == 0)
        if due and isinstance(self.training_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
            sync_envs_normalization(self.training_env, self.eval_env)
            self.eval_env.training = False
            self.eval_env.norm_reward = False
        return super()._on_step()

class TensorboardMetricsCallback(BaseCallback):
    def __init__(self, smoothing: int = 100, verbose: int = 0):
        super().__init__(verbose)
        # self.buf = {k: deque(maxlen=smoothing) for k in ["distance", "angle_to_normal_deg","r_dist","r_ang"]}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "final_metrics" in info:
                for k, v in info["final_metrics"].items():
                    self.logger.record(f"metrics/final_{k}", float(v))

        return True

def make_env(port):
    env = DrillRivetEnv(port)
    env = TimeLimit(env, max_episode_steps=250)
    return lambda: Monitor(env)

def main():
    train_env = DummyVecEnv([make_env(23000)])
    # train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_reward=10.0)

    os.makedirs("trained_models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    action_dim = train_env.action_space.shape[0]
    model = SAC(
        "MlpPolicy", train_env,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=500000,
        tau=0.005, gamma=0.99,
        ent_coef="auto_1.0",
        # target_entropy=-(0.8 * action_dim),
        target_entropy=-(action_dim),
        learning_starts=20000,
        train_freq=(1, "step"),
        gradient_steps=1,
        tensorboard_log="./logs/",
        verbose=1,
        device="auto",
    )
    # policy_kwargs=dict(net_arch=dict(pi=[256,256], qf=[256,256]), log_std_init=-1.0)

    ckpt_cb = CheckpointCallback(
        save_freq=50000,
        save_path="./ckpt/err",
        name_prefix="ckpt",
        save_vecnormalize=True,
        save_replay_buffer=True,
        verbose=1
    )
    tb_cb = TensorboardMetricsCallback(smoothing=100)

    total_steps = 1000000
    model.learn(total_timesteps=total_steps, callback=[ckpt_cb, tb_cb], tb_log_name="err", log_interval=10)

    # 保存模型与 VecNormalize 统计量
    model.save("trained_models/final_sac_model_err")
    if isinstance(train_env, VecNormalize):
        train_env.save("logs/vecnorm_err.pkl")

    # env.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
