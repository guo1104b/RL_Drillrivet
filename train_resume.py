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
    
    if not isinstance(src, VecNormalize) or not isinstance(dest, VecNormalize):
        raise TypeError("Both envs must be VecNormalize instances")

    dest.obs_rms.mean = src.obs_rms.mean.copy()
    dest.obs_rms.var = src.obs_rms.var.copy()
    dest.obs_rms.count = src.obs_rms.count

class CrossProcEvalCallback(EvalCallback):
    def _on_step(self) -> bool:
        
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
            if "final_metrics" in info:
                for k, v in info["final_metrics"].items():
                    self.logger.record(f"metrics/final_{k}", float(v))

        # for k, dq in self.buf.items():
        #     if dq:
        #         self.logger.record(f"metrics/{k}_ma", float(np.mean(dq)))
        return True

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

train_env_raw = DummyVecEnv([make_env(23000)])

train_env = VecNormalize.load(vec_path, train_env_raw)  
train_env.training = True
train_env.norm_reward = True
train_env.clip_reward=10.0

# eval_env = VecNormalize.load(vec_path, eval_env_raw)     
# eval_env.training = False
# eval_env.norm_reward = False

model = SAC.load(
    ckpt_path, env=train_env, device="auto", print_system_info=True)
    # custom_objects={"learning_rate": 1e-4, "ent_coef": "auto_1.0"}, print_system_info=True)
print(f"Loading model: {ckpt_path}")
model.learning_starts = int(model.num_timesteps)
action_dim = model.action_space.shape[0]
model.target_entropy = -(0.3 * action_dim)  

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

MORE_STEPS = 1000000
model.learn(
    total_timesteps=MORE_STEPS,
    reset_num_timesteps=False,   
    callback=[ckpt_cb, tb_cb],
    log_interval=10,
    tb_log_name="c1_resume"
)

model.save("trained_models/final_sac_model_resumed")
train_env.save("logs/vecnorm_resumed.pkl")
print("Resumed training finished.")
