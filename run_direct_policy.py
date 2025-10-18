#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Direct deterministic policy control without Gym wrappers.

- You script Bill's actions yourself in CoppeliaSim (or via vrep.ActionRunner).
- This script only:
    * connects to VrepInterface
    * collects observations like in DrillRivetEnv._get_obs
    * (optionally) normalizes obs using provided stats (npz with 'mean','var')
    * loads a trained SB3 model
    * queries deterministic actions and drives the robot at each sim step

Usage:
  python run_direct_policy.py --ckpt ./models/sac_last.zip --target 3 --port 23000 --steps 250
  python run_direct_policy.py --ckpt ./models/sac_last.zip --target 3 --stats-npz ./obs_stats.npz --steps 300
  python run_direct_policy.py --ckpt ./models/td3.zip --algo td3 --target 3 --sleep 0.0

Stats file (optional):
  np.savez('obs_stats.npz', mean=mean_vec, var=var_vec)
  where mean/var are length-32 arrays matching the observation ordering below.

Observation ordering (dim=32):
  [ q(6), qdot(6), p_ee(3), target_pose(3), theta(1), theta_y(1),
    p_h(2), u_h(1), p_hand(3), u_hand(1), phase_oh(5) ]

Success condition (mirrors your env defaults):
  d < 0.005  and  |theta| < 0.1 rad  and  |theta_y| < 0.05 rad
"""

import argparse
import time
import math
import numpy as np
import pickle

from stable_baselines3 import SAC, PPO, TD3
from vrep2 import VrepInterface


# with open('trained_models/final.pkl', 'rb') as f:
#     vn = pickle.load(f)
# # vn 是 VecNormalize 对象
# mean = vn.obs_rms.mean
# var = vn.obs_rms.var
# np.savez('obs_stats_final.npz', mean=mean, var=var)
# print("✅ 导出成功：obs_stats_final.npz")

def phase_onehot(pid: int, n_phase: int = 5) -> np.ndarray:
    oh = np.zeros(n_phase, dtype=np.float32)
    if 0 <= int(pid) < n_phase:
        oh[int(pid)] = 1.0
    return oh

def build_obs(v: VrepInterface, target_idx: int, prev_body_xy: np.ndarray, prev_hand_pos: np.ndarray, dt: float,
              u_h: float = 1.0, u_hand: float = 1.0):
    """
    Collects observation in the same order as DrillRivetEnv._get_obs.
    Returns: obs(32,), d (distance to target), theta(rad), theta_y(rad),
             new_body_xy(2,), new_hand_pos(3,)
    """
    # --- Kinematics ---
    q = np.asarray(v.get_joint_positions(), dtype=np.float32)             # (6,)
    qdot = np.asarray(v.get_joint_velocities(), dtype=np.float32)         # (6,)

    target_pose = np.asarray(v.targets_pos[target_idx - 1], dtype=np.float32)  # (3,)
    place_pose  = np.asarray(v.places_pos[target_idx - 1], dtype=np.float32)   # (3,)

    ee_pos = np.asarray(v.get_object_position(v.ee), dtype=np.float32)    # (3,)
    p_ee = ee_pos - target_pose                                           # (3,)

    # --- Human / Hand pose ---
    bill_pos = np.asarray(v.get_object_position(v.Bill), dtype=np.float32)      # (3,)
    body_xy = bill_pos[:2]
    p_h = body_xy - place_pose[:2]                                         # (2,)

    hand_pos = np.asarray(v.get_object_position(v.Hand), dtype=np.float32) # (3,)
    p_hand = hand_pos - place_pose                                         # (3,)

    # --- Orientation angles ---
    _, ang_ee_deg, _, ang_wy_deg = v.ee_to_normal_angle(target_pose.tolist(), v.center_pos, v.axis_dir, v.tcp)
    theta = float(abs(np.deg2rad(ang_ee_deg)))     # rad
    theta_y = float(abs(np.deg2rad(ang_wy_deg)))   # rad

    u_h = np.array([u_h], dtype=np.float32)
    u_hand = np.array([u_hand], dtype=np.float32)

    # --- Phase onehot ---
    try:
        pid = int(v.runner.get_phase_id())
    except Exception:
        pid = -1
    ph_oh = phase_onehot(pid, n_phase=5)

    # --- Pack obs ---
    obs = np.concatenate([
        q, qdot, p_ee, target_pose,
        np.array([theta], dtype=np.float32),
        np.array([theta_y], dtype=np.float32),
        p_h.astype(np.float32),
        u_h,
        p_hand.astype(np.float32),
        u_hand,
        ph_oh.astype(np.float32)
    ]).astype(np.float32)

    d = float(np.linalg.norm(p_ee))
    return obs, d, theta, theta_y, body_xy.copy(), hand_pos.copy()

def maybe_normalize(obs: np.ndarray, stats):
    if stats is None:
        return obs
    mean = np.asarray(stats.get('mean'), dtype=np.float32)
    var  = np.asarray(stats.get('var'), dtype=np.float32)
    assert mean.shape == obs.shape and var.shape == obs.shape, f"stats shape {mean.shape}/{var.shape} != obs {obs.shape}"
    return (obs - mean) / np.sqrt(var + 1e-8)

def move_with_direct_policy(v,target,stats,model,prev_body_xy, prev_hand_pos, u=1.0, dt=0.05, clip=0.05, steps=250):
    try:
        for t in range(1, steps+1):
            obs, d, theta, theta_y, body_xy, hand_pos = build_obs(
                v, target, prev_body_xy, prev_hand_pos, dt,u,u
            )
            prev_body_xy, prev_hand_pos = body_xy, hand_pos

            obs_in = maybe_normalize(obs, stats)
            action, _ = model.predict(obs_in, deterministic=True)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            if np.isscalar(clip) and clip > 0:
                action = np.clip(action, -clip, clip)

            v.move_robot(action.tolist())

            # advance simulation one step; tick runner if present
            try:
                if v.runner.is_busy() or v.runner.queue:
                    v.runner.tick()
            except Exception:
                pass
            v.sim.step()

            if d <= 0.01:
                break

            # success check
            # success = (d < dist_thr) and (abs(theta) < theta_thr) and (abs(theta_y) < theta_y_thr)
            # if t % 10 == 0 or success:
            if t % 10 == 0:
                 print(f"[t={t:03d}] d={d:.4f}, theta={theta:.3f}, theta_y={theta_y:.3f}, action={np.round(action,4)}")
            # if success:
            #     print(f"[SUCCESS] Reached: d<{dist_thr}, |theta|<{theta_thr}, |theta_y|<{theta_y_thr}")
            #     break

        v.stop_joint_velocities()
    finally:
        # keep simulation running for inspection; comment next line if you prefer to stop sim
        # v.stop()
        pass
    return prev_body_xy, prev_hand_pos

def main():

    model = SAC.load("./trained_models/final.zip", device="auto")
    model0 = SAC.load("./ckpt_550000_steps.zip", device="auto")

    # Load stats if provided
    stats = None
    stats_npz = './obs_stats_final.npz'
    if stats_npz:
        try:
            data = np.load(stats_npz)
            stats = {'mean': data['mean'], 'var': data['var']}
            print(f"[Info] Loaded obs stats from {stats_npz}")
        except Exception as e:
            print(f"[Warn] Failed to load stats: {e}. Running without normalization.")
    stats_npz0 = './obs_stats0.npz'
    data0 = np.load(stats_npz0)
    stats0 = {'mean': data0['mean'], 'var': data0['var']}

    # Connect to CoppeliaSim
    v = VrepInterface(23000, stepping=True)
    v.start(wait=1.0)
    v.stop_joint_velocities()

    # thresholds consistent with your env
    dist_thr = 0.01
    theta_thr = 0.1
    theta_y_thr = 0.1
    dt = 0.05  # control period
    clip = 0.05

    prev_body_xy = None
    prev_hand_pos = None

    present_target = 3
    ang_deg, dist = v.reset_bill([present_target])
    v.move_bill_init(ang_deg, dist, present_target, 1)
    prev_body_xy, prev_hand_pos = move_with_direct_policy(v,present_target,stats0,model0,prev_body_xy, prev_hand_pos, u=1.0,steps=220)

    v.reset_robot([0.3574, 0.3107, -0.2264, -0.1216, 1.7415, -0.3360])

    v.move_bill_handsdown(pls=4, k=1)

    # v.reset_bill2(3)

    # v.move_to_position(v.ik[2])
    # time.sleep(1)

    v.move_bill(pls=4, k=1)
    present_target = 4
    v.stop_joint_velocities()
    prev_body_xy, prev_hand_pos = move_with_direct_policy(v,present_target,stats,model,prev_body_xy, prev_hand_pos, u=1.0)



    time.sleep(1)
    present_target = 5
    v.stop_joint_velocities()
    v.move_bill_handsdown(pls=present_target, k=1)
    v.move_bill2(pls0=4, pls=present_target, k=1)

    prev_body_xy, prev_hand_pos = move_with_direct_policy(v,present_target,stats,model,prev_body_xy, prev_hand_pos, u=1.0)


    time.sleep(1)
    present_target = 6
    v.stop_joint_velocities()
    v.move_bill_handsdown(pls=present_target, k=1)
    v.move_bill2(pls0=5, pls=present_target, k=1)
    prev_body_xy, prev_hand_pos = move_with_direct_policy(v,present_target,stats,model,prev_body_xy, prev_hand_pos, u=1.0)

if __name__ == "__main__":
    main()
