# r_dist权重大一点
# Environment in gymnasium style
# -*- coding: utf-8 -*-
import time
from typing import Optional, Tuple, Dict
import gymnasium as gym
import numpy as np
from vrep2 import VrepInterface

class DrillRivetEnv(gym.Env):
    """
    A阶段：只训练“到就位点 + 对孔法向对齐 + 人机同步 + 机身不碰撞”。

    obs = concat([q(6), qdot(6), p_ee(3), target_pose(3), theta(1), theta_y(1), p_h(2), u_h(1), p_hand(3), u_hand(1)])

    action = joint velocity ∈ [-0.1, 0.1]
    """
    def __init__(self, port):
        super().__init__()

        self.v = VrepInterface(port)
        self.q0 = self.v.get_joint_positions() # 需要改
        self.v.start()
        self.target = 3
        self.u_h = 1
        self.u_hand = 1
        self.prev_qdot = np.zeros_like(self.v.get_joint_velocities())
        self.prev_action = np.zeros(6, dtype=float)
        self.dt = 0.05
        self.t_human_reach = 0
        self.t_robot_reach = 0
        self.maxSteps = 250
        self.currentStep = 0
        self.dist_thr = 0.01
        self.theta_thr = 0.1
        self.theta_y_thr = 0.1

        self.total_steps = 0
        self.k_curriculum_steps = 200000

        self.phase_names = ['step_forward', 'rotate', 'move_hand', 'reset_hand', 'sidestep']
        self.n_phase = len(self.phase_names)

        target_pose = self.v.targets_pos[self.target - 1]
        ee_pos0 = self.v.get_object_position(self.v.ee)
        self.prev_d = np.linalg.norm(np.array(ee_pos0) - np.array(target_pose))

        _, ang_ee, _, ang_wy = self.v.ee_to_normal_angle(target_pose, self.v.center_pos, self.v.axis_dir, self.v.tcp)
        self.prev_theta = float(abs(np.deg2rad(ang_ee)))
        self.prev_theta_y = float(abs(np.deg2rad(ang_wy)))

        self.action_space = gym.spaces.Box(low=-0.05, high=0.05, shape=(6,), dtype=np.float32)
        # 观测空间维度：6 + 6 + 3 + 3 + 3+ 1 + 3 + 4 =
        # q, qdot, p_ee(relative), target_pose, theta,theta_y, p_h二维(relative), u_h, p_hand(relative), u_hand
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32)

    # ----------------- Gym API -----------------
    def reset(self, seed= None, options= None):
        try:
            self.v.stop_joint_velocities()
            self.v.reset_robot(self.q0)
            # self.target = 6
            self.target = self.sample_target_curriculum()
            # self.target = int(np.random.choice([4, 5, 6], p=np.array([0.30, 0.30, 0.40], dtype=float)))
            self.v.reset_bill(self.target) # self.target = np.random.randint(1, 7)
            # k = self.sample_k_curriculum()
            k = int(np.random.choice([1, 2, 4], p=np.array([0.30, 0.40, 0.30], dtype=float)))
            self.u_h = k
            self.u_hand = k
            # k=int(np.random.choice([1, 5, 20], p=np.array([0.50, 0.25, 0.25], dtype=float)))
            self.v.move_bill(self.target,k)

            self.prev_qdot = np.zeros_like(self.v.get_joint_velocities())
            self.prev_action = np.zeros(6, dtype=float)
            target_pose = self.v.targets_pos[self.target - 1]
            ee_pos0 = self.v.get_object_position(self.v.ee)
            self.prev_d = np.linalg.norm(np.array(ee_pos0) - np.array(target_pose))
            self.initial_distance = self.prev_d
            _, ang_ee, _, ang_wy = self.v.ee_to_normal_angle(target_pose, self.v.center_pos, self.v.axis_dir,
                                                             self.v.tcp)
            self.prev_theta = float(abs(np.deg2rad(ang_ee)))
            self.prev_theta_y = float(abs(np.deg2rad(ang_wy)))


            self.t_human_reach = 0
            self.t_robot_reach = 0
            self.currentStep = 0
            obs, _, _ = self._get_obs()
            obs = obs.astype(np.float32)
            return obs, {}
        except Exception:
            print("Stop and start simulation again!")
            self.v.stop()
            self.v.start()
            # self.target = 6
            # self.target = int(np.random.choice([4, 5, 6], p=np.array([0.30, 0.30, 0.40], dtype=float)))
            self.target = self.sample_target_curriculum()

            self.v.reset_bill(self.target)
            # k = self.sample_k_curriculum()
            k = int(np.random.choice([1, 2, 4], p=np.array([0.30, 0.40, 0.30], dtype=float)))
            self.u_h = k
            self.u_hand = k
            self.v.move_bill(self.target, k)
            self.prev_qdot = np.zeros_like(self.v.get_joint_velocities())
            self.prev_action = np.zeros(6, dtype=float)
            target_pose = self.v.targets_pos[self.target - 1]
            ee_pos0 = self.v.get_object_position(self.v.ee)
            self.prev_d = np.linalg.norm(np.array(ee_pos0) - np.array(target_pose))
            self.initial_distance = self.prev_d
            _, ang_ee, _, ang_wy = self.v.ee_to_normal_angle(target_pose, self.v.center_pos, self.v.axis_dir,
                                                             self.v.tcp)
            self.prev_theta = float(abs(np.deg2rad(ang_ee)))
            self.prev_theta_y = float(abs(np.deg2rad(ang_wy)))


            self.t_human_reach = 0
            self.t_robot_reach = 0
            self.currentStep = 0
            obs, _, _ = self._get_obs()
            obs = obs.astype(np.float32)
            return obs, {}

    def step(self, action):
        self.currentStep += 1
        self.total_steps += 1

        self.v.move_robot(action)
        if self.v.runner.is_busy() or self.v.runner.queue:
            ret = self.v.runner.tick()
            if ret and ret[0] == "finished" and ret[1] == "Bill_MoveHand_Done":
                self.t_human_reach = self.currentStep
                # print(f"self.t_human_reach is {self.t_human_reach}")
            self.v.sim.step()
        else:
            self.v.sim.step()

        obs, ang_ee, ang_wy = self._get_obs()
        q, qdot, p_ee, target_pose, theta, theta_y, p_h, u_h, p_hand, u_hand = self.parse_state(obs)
        reward, terminated, truncated, info = self.compute_reward_terminate(action, p_ee, p_hand, theta, theta_y, self.currentStep)

        # print(f"r_dist, r_ang, r_collision, r_smooth, r_reach, r_sync: {info['reward']}")
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        #q, qdot, p_ee(relative), ori_ee, p_target, p_h(relative 2 dimension), u_h, p_hand(relative), u_hand
        q = np.array(self.v.get_joint_positions())
        qdot = np.array(self.v.get_joint_velocities())
        target_pose = self.v.targets_pos[self.target - 1] # 之后要修改
        place_pose = self.v.places_pos[self.target - 1]  # 之后要修改
        ee_pos = self.v.get_object_position(self.v.ee)
        p_ee = np.array(ee_pos) - np.array(target_pose)
        # ori_ee = np.array(self.v.get_object_orientation(self.v.ee))
        bill_pos = self.v.get_object_position(self.v.Bill)
        p_h = np.array(bill_pos[:2]) - np.array(place_pose[:2])
        hand_pos = self.v.get_object_position(self.v.Hand)

        p_hand = np.array(hand_pos) - np.array(place_pose)

        _, ang_ee, _, ang_wy = self.v.ee_to_normal_angle(target_pose, self.v.center_pos, self.v.axis_dir, self.v.tcp)

        theta = float(abs(np.deg2rad(ang_ee)))
        theta_y = float(abs(np.deg2rad(ang_wy)))
        u_h = self._vec(self.u_h)
        u_hand = self._vec(self.u_hand)
        theta = self._vec(theta)
        theta_y = self._vec(theta_y)

        pid = int(self.v.runner.get_phase_id())
        phase_oh = self.phase_onehot(pid)
        # print(f"phase_oh is {phase_oh}")

        state = np.concatenate([q, qdot, p_ee, np.array(target_pose), theta, theta_y,
                                p_h, u_h, p_hand, u_hand, phase_oh]).astype(np.float32)
        return np.round(state, decimals=4), ang_ee, ang_wy

    def render(self):
        pass

    def close(self):
        try:
            self.v.stop()
        except Exception:
            pass

    # ----------------- Internal calculation tools -----------------
    def _vec(self, x, dtype=float):
        return np.atleast_1d(np.asarray(x, dtype=dtype)).reshape(-1)

    def compute_reward_terminate(self, action, p_ee, p_hand, theta, theta_y, currentStep):

        terminated = False
        truncated = False
        # -------------- Distance ------------------
        d = float(np.linalg.norm(p_ee))  # 与目标的欧氏距离
        dh = float(np.linalg.norm(p_hand))
        derr = abs(d-dh)
        # print(f"d is {d}, dh is {dh}, derr is {derr}")

        # if d > 0.1:
        #     r_dist = (self.prev_d - d) - 2.0 * derr
        # elif d > 0.05:
        #     r_dist = 0.5 * np.exp(- 10 * derr) + 2 * (self.prev_d - d)
        # else:
        #     r_dist = 1.0 * np.exp(- 50 * derr) + 2 * (self.prev_d - d)
        r_dist = (self.prev_d - d) - 2.0 * derr
        self.prev_d = d
        # r_dist = - 2.0 * derr

        if d > 2.0 * self.initial_distance:
            r_dist -= 10.0
            terminated = True

        # ---------- Orientation alignment---------
        w_gate2 = 0.6 + 0.4 * self.smoothstep(d, hi=0.20, lo=0.02)
        # w_gate2 = 1.0 - 0.6 * self.smoothstep(d, hi=0.30, lo=0.08)
        theta = float(theta)
        theta_y = float(theta_y)
        # r_ang = w_gate2 * (0.5 * self.bowl(theta, np.deg2rad(15)) +
        #                    0.3 * self.bowl(theta_y, np.deg2rad(12))) - 0.02 * (theta**2 + theta_y**2)
        # r_ang += (self.prev_theta - theta) + 0.5 * (self.prev_theta_y - theta_y)
        # self.prev_theta, self.prev_theta_y = theta, theta_y
        r_ang = - 2 * (theta**2 + theta_y**2) * w_gate2
        # 近端调姿态
        # r_ang = w_gate2 * (0.6 * bowl(theta, np.deg2rad(12)) + 0.4 * bowl(theta_y, np.deg2rad(15)))

        # ----------- Collision penalty -----------
        collided = bool(self.v.if_collision())
        r_collision = -100.0 if collided else 0.0

        # ------------ Suppress jitter ------------
        adot = (np.asarray(action, float) - self.prev_action) / self.dt
        # w_gate3 = 0.7 + 0.3 * (1.0 - self.smoothstep(d, hi=0.8, lo=0.2)) #改上下限或者去掉门限
        w_gate3 = 0.7 + 0.3 * self.smoothstep(d, hi=0.2, lo=0.02)
        r_smooth = - 0.05 * float(np.dot(adot, adot)) * w_gate3
        self.prev_action = np.asarray(action, float)
        # print(f"r_smooth every step: {r_smooth}, with gate of {w_gate3}, with dist of {d}")
        # print(f"r_smooth every step: {r_smooth}, r_dist every step {r_dist}, r_ang every step {r_ang}")

        # ------------ Synchronization ------------
        r_reach = 0
        r_sync = 0
        if self.t_human_reach != 0:
            # r_sync = 0.01 * np.exp(-currentStep + self.t_human_reach) - 0.01
            r_sync = -0.001 * (currentStep - self.t_human_reach)
        if collided:
            terminated = True
        elif d < self.dist_thr and abs(theta) < self.theta_thr and abs(theta_y) < self.theta_y_thr:
            self.t_robot_reach = currentStep
            r_reach = 100
            terminated = True
            qt = self.v.get_joint_positions()
            print(f"When target is {self.target}, qt is {qt}")
            # Bill_pose = self.v.sim.getObjectPose(self.v.Bill, -1)
            # print(f"Bill_pose is {Bill_pose}")
            # info["is_success"] = True
            if self.t_human_reach != 0:
                r_sync = 100 - 0.5 * abs(self.t_robot_reach - self.t_human_reach)
            else:
            #     r_sync = 50 * (np.exp(-2*abs(float(np.linalg.norm(p_hand))))-1)
                r_sync = 10

        # ----------- Total reward ----------
        reward = r_dist + r_ang + r_collision + r_smooth + r_reach + r_sync
        reward = np.round(reward, decimals=4)
        # info["reward"] = [r_dist, r_ang, r_collision, r_smooth, r_reach, r_sync]

        info = {
            "collided": collided,
            "is_success": bool(r_reach > 0),
            "reward": [float(r_dist), float(r_ang), float(r_collision), float(r_smooth), float(r_reach), float(r_sync)],
            "metrics": {
                "distance": float(d),
                "angle_to_normal": float(theta),
                "angle_to_y_normal": float(theta_y),
                "r_dist": float(r_dist),
                "r_ang": float(r_ang), "r_smooth": float(r_smooth)
            }
        }

        if self.currentStep >= self.maxSteps:
            truncated = True
            reward -= 10.0

        if terminated or truncated:
            info["final_metrics"] = dict(info["metrics"])

        return float(reward), bool(terminated), bool(truncated), info

    def smoothstep(self, x, hi, lo):
        t = np.clip((hi - x) / (hi - lo), 0.0, 1.0)
        return t * t * (3 - 2 * t)

    def bowl(self, a, a_star):  # a<=a_star 给正值，之外为0
        x = float(a) / float(a_star)
        return max(1.0 - x * x, 0.0)

    def phase_onehot(self, pid: int) -> np.ndarray:
        """
        将整数 phase_id 映射为 one-hot；当 pid==-1（无活动阶段）时返回全零。
        """
        oh = np.zeros(self.n_phase, dtype=np.float32)
        if 0 <= int(pid) < self.n_phase:
            oh[int(pid)] = 1.0
        return oh

    def parse_state(self, state):
        idx = 0
        q = state[idx:idx + 6]; idx += 6
        qdot = state[idx:idx + 6]; idx += 6
        p_ee = state[idx:idx + 3]; idx += 3
        # ori_ee = state[idx:idx + 3]; idx += 3
        target_pose = state[idx:idx + 3]; idx += 3
        theta = state[idx:idx + 1]; idx += 1
        theta_y = state[idx:idx + 1]; idx += 1
        p_h = state[idx:idx + 2]; idx += 2
        u_h = state[idx:idx + 1]; idx += 1
        p_hand = state[idx:idx + 3]; idx += 3
        u_hand = state[idx:idx + 1]; idx += 1
        return q, qdot, p_ee, target_pose, theta, theta_y, p_h, u_h, p_hand, u_hand

    def sample_k_curriculum(self):
        # 进度 0→1（达到 k_curriculum_steps 后封顶）
        nearest = (self.total_steps // 30000) * 30000
        prog = min(1.0, nearest / float(self.k_curriculum_steps))
        # 起始分布：几乎都是 1；目标分布：5/20 占大头
        p0 = np.array([0.80, 0.10, 0.10], dtype=float)  # 初始分布
        p1 = np.array([0.40, 0.30, 0.30], dtype=float)  # 目标分布
        p = (1 - prog) * p0 + prog * p1
        p = p / p.sum()

        return int(np.random.choice([2, 1, 4], p=p))
    
    def sample_target_curriculum(self):
        if self.total_steps < 2 * self.k_curriculum_steps:
            return 6
        else:
            nearest2 = (self.total_steps // 50000) * 50000
            prog = min(1.0, nearest2 / float(2*self.k_curriculum_steps) - 1.0)
            # p0 = np.full(6, 0.05, dtype=float)
            # p0[2] = 0.75
            # p1 = np.full(6, 0.16, dtype=float)
            # p1[2] = 0.2
            # p = (1 - prog2) * p0 + prog2 * p1
            p0 = np.array([0.10, 0.10, 0.80], dtype=float)  # 初始分布
            p1 = np.array([0.30, 0.30, 0.40], dtype=float)  # 目标分布
            p = (1 - prog) * p0 + prog * p1
            p = p / p.sum()
            # return int(np.random.choice(np.arange(1, 7), p=p))
            return int(np.random.choice([4, 5, 6], p=p))

if __name__ == "__main__":
    # train_stub.py
    # import numpy as np
    # from env import DrillRivetEnv

    env = DrillRivetEnv(23000)
    obs = env.reset()
    done = False
    trunc = False
    # while not (done or trunc):
    for _ in range(250):
        a = env.action_space.sample()  # 随机动作测试
        obs, r, done, trunc, info = env.step(a)
        # print(r, info)
    env.close()
