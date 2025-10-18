import os
import numpy as np
from collections import deque

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import SACPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from envC import DrillRivetEnv

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
        if isinstance(self.training_env, VecNormalize) and isinstance(self.eval_env, VecNormalize):
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
            # 兼容两种写法：优先用 info["metrics"]，否则用扁平键
            # metrics = info.get("metrics") or {
            #     k: info[k] for k in ("distance", "angle_to_normal_deg","r_dist","r_ang") if k in info
            # }
            # for k, v in metrics.items():
            #     if k in self.buf:
            #         self.buf[k].append(float(v))

            # 回合结束时的最终指标：画到 metrics/final_*
            if "final_metrics" in info:
                for k, v in info["final_metrics"].items():
                    self.logger.record(f"metrics/final_{k}", float(v))

        # 每步滑动平均：画到 metrics/{key}_ma
        # for k, dq in self.buf.items():
        #     if dq:
        #         self.logger.record(f"metrics/{k}_ma", float(np.mean(dq)))
        return True

# ==== 0) 指定要恢复的 checkpoint 步数 ====
RESUME_STEP = 1000000
CKPT_DIR = "./ckpt/c1"

ckpt_path = f"{CKPT_DIR}/ckpt_{RESUME_STEP}_steps.zip"
vec_path  = f"{CKPT_DIR}/ckpt_vecnormalize_{RESUME_STEP}_steps.pkl"
rb_path   = f"{CKPT_DIR}/ckpt_replay_buffer_{RESUME_STEP}_steps.pkl"

def make_env(port):
    env = DrillRivetEnv(port)
    env = TimeLimit(env, max_episode_steps=250)
    return lambda: Monitor(env)

def lr_schedule(progress_remaining: float) -> float:
    start, end = 3e-4, 1e-4
    return end + (start - end) * progress_remaining

# ==== 1) 先建“未归一化”的 env，再加载 VecNormalize 统计 ====
train_env_raw = DummyVecEnv([make_env(23000)])

# 用与 ckpt 同步的统计覆盖 train/eval 两个 VecNormalize
train_env = VecNormalize.load(vec_path, train_env_raw)   # 训练：统计会继续更新
train_env.training = True
train_env.norm_reward = True
train_env.clip_reward=10.0

# eval_env = VecNormalize.load(vec_path, eval_env_raw)     # 评估：冻结统计，不更新
# eval_env.training = False
# eval_env.norm_reward = False

# ==== 2) 加载模型（绑定到已经加载统计的 train_env）====
model = SAC.load(
    ckpt_path, env=train_env, device="auto", print_system_info=True)
    # custom_objects={"learning_rate": 1e-4, "ent_coef": "auto_1.0"}, print_system_info=True)
print(f"Loading model: {ckpt_path}")
# === 目标熵改回更稳的默认：-action_dim ===
model.learning_starts = int(model.num_timesteps)
action_dim = model.action_space.shape[0]
model.target_entropy = -(0.3 * action_dim)  # 可试 0.5~0.6

# ==== 3) 恢复 Replay Buffer（可选但强烈推荐）====
# 若不存在该文件，可跳过这一步，模型也能继续训练，只是“热启动”变慢
try:
    model.load_replay_buffer(rb_path)
    print(f"[Resume] Loaded replay buffer from {rb_path}")
except FileNotFoundError:
    print(f"[Resume] Replay buffer not found at {rb_path}, continue without it.")

ckpt_cb = CheckpointCallback(
    save_freq=50000,
    save_path="./ckpt/c1_resume",
    name_prefix="ckpt",
    save_vecnormalize=True,
    save_replay_buffer=True,
    verbose=1
)
tb_cb = TensorboardMetricsCallback(smoothing=100)

# ==== 5) 继续训练（**一定**要 reset_num_timesteps=False）====
MORE_STEPS = 1000000
model.learn(
    total_timesteps=MORE_STEPS,
    reset_num_timesteps=False,   # 保持全局步数连续，TensorBoard/学习率调度都不乱
    callback=[ckpt_cb, tb_cb],
    log_interval=10,
    tb_log_name="c1_resume"
)

# ==== 6) 如需再次保存最终模型（可选）====
model.save("trained_models/final_sac_model_resumed")
train_env.save("logs/vecnorm_resumed.pkl")
print("Resumed training finished.")
