# CoppeliaSim ZMQ Remote API interface for Gym-style env.
# vrep.py
# -*- coding: utf-8 -*-
from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import time
import math
import numpy as np
from runner import ActionRunner

class VrepInterface:
    def __init__(self, port, stepping: bool = True):
        # self.client = RemoteAPIClient()
        self.client = RemoteAPIClient('localhost', port)
        self.sim = self.client.require('sim')
        if not self.sim:
            print('V-REP remote API server connection failed. Is V-REP running?')
        print(f'Connected to Remote API Server, port number is {port}')
        self.setup_handles()
        if stepping:
            self.sim.setStepping(True)
        self.runner = ActionRunner(self.sim)
        self.collision = False


    def setup_handles(self):
        self.Bill = self.sim.getObject('/Bill')
        self.Bill_init = self.sim.getObjectPose(self.Bill, -1)
        # self.Hand = self.sim.getObject('/gt_hand')
        self.Hand = self.sim.getObject('/rightHand_tip')
        self.fuselage = self.sim.getObject('/Fuselage')
        self.center_pos = self.sim.getObjectPosition(self.fuselage, -1)
        self.axis_dir = [0, 1, 0]  # 圆柱轴沿世界 y
        self.tcp = self.sim.getObject('/drill')

        self.Robot = self.sim.getObject('/kuka_kr120')
        self.Robot_controller = self.sim.getObject('/kuka_kr120/script')
        self.ee = self.sim.getObject('/TCP_dummy')

        joint_names = ["/joint1", "/joint2", "/joint3", "/joint4", "/joint5", "/joint6"]
        self. joints = [self.sim.getObject(name) for name in joint_names]
        tgt_names = ['/target1', '/target2', '/target3', '/target4', '/target5', '/target6']
        self.targets = [self.sim.getObject(name) for name in tgt_names]
        self.targets_pos = [self.sim.getObjectPosition(target) for target in self.targets]
        place_names = ['/place1', '/place2', '/place3', '/place4', '/place5', '/place6']
        self.places = [self.sim.getObject(name) for name in place_names]
        self.places_pos = [self.get_object_position(h) for h in self.places]
        self.place_init = self.sim.getObject('./place_init')

    def start(self, wait: float = 1.0):
        self.sim.startSimulation()
        if wait > 0:
            time.sleep(wait)

        self.stop_joint_velocities()

        self.runner.enqueue(do_signal='Bill_do_InitHand', done_signal='Bill_InitHand_Done', timeout=50.0)
        while self.runner.is_busy() or self.runner.queue:
            self.runner.tick()
            self.sim.step()

    def pause(self):
        self.sim.pauseSimulation()

    def stop(self, wait: float = 0.5):
        self.sim.stopSimulation()
        if wait > 0:
            time.sleep(wait)

    def if_collision(self):
        if self.sim.getIntegerSignal('Collision') or self.sim.getIntegerSignal('Collision3'):
            print("Collision!")
            self.collision = True
        return self.collision

    def reset_bill(self, pls):
        self.clear_runner_queue()
        self.runner.enqueue(do_signal='Bill_do_ResetHand', done_signal='Bill_ResetHand_Done', timeout=10.0)

        pose = list(self.Bill_init)  # [x,y,z, qx,qy,qz,qw]
        x0, y0, z0 = float(pose[0]), float(pose[1]), float(pose[2])
        new_x = x0 + float(np.random.uniform(-0.1, 0.1))
        new_y = y0 + float(np.random.uniform(-0.1, 0.1))
        new_z = z0
        pose[0], pose[1], pose[2] = new_x, new_y, new_z
        self.sim.setObjectPose(self.Bill, pose)

        tid = int(pls[0])
        tgt_h = self.places[tid - 1]
        tgt_pos = self.sim.getObjectPosition(tgt_h, -1)
        tx, ty = float(tgt_pos[0]), float(tgt_pos[1])
        dx, dy = (tx - new_x - 0.25), (ty - new_y)
        ang_rad = math.atan2(dy, dx)
        ang_deg = math.degrees(ang_rad)
        dist = math.sqrt(dx**2 + dy**2)

        # self.sim.setObjectPose(self.Bill, self.Bill_init)
        # ang_deg, dist = 0, 0.2

        self.runner.enqueue(
            do_signal='Bill_do_Rotate',
            done_signal='Bill_Rotate_Done',
            ints={'Bill_Rotate_K': 10, 'Bill_Rotate_Angle': ang_deg},
        )
        while self.runner.is_busy() or self.runner.queue:
            self.runner.tick()
            self.sim.step()
        return ang_deg, dist

    def reset_robot(self, q):

        for j in self.joints:
            self.sim.setJointTargetVelocity(j, 0)
        self.sim.step()

        for j, qq in zip(self.joints, q):
            self.sim.setObjectInt32Param(j, self.sim.jointintparam_dynctrlmode, self.sim.jointdynctrl_position)
            self.sim.setJointTargetPosition(j, qq)

        while True:
            self.sim.step()
            err = 0.0
            for j, qq in zip(self.joints, q):
                err = max(err, abs(self.sim.getJointPosition(j) - qq))
            if err <= 1e-3:
                break

        for j in self.joints:
            self.sim.setObjectInt32Param(j, self.sim.jointintparam_dynctrlmode, self.sim.jointdynctrl_velocity)
            self.sim.setJointTargetVelocity(j, 0)

        self.sim.setIntegerSignal('Reset_Trace', 1)
        self.collision = False
        self.sim.step()
        return True

    def move_bill(self, ang, dist, pls, kk):
        # In phase A only steps forward and moves hand
        # In phase B will add actions Rotate and StepSideways, change K
        # phase_id={stepfwd:0, rotate:1, movehand:2, resethand:3, stepside:4}
        p1 = int(pls[0]) if len(pls) >= 1 else 0
        p2 = int(pls[1]) if len(pls) >= 2 else 0
        k1 = int(kk[0]) if len(kk) >= 1 else 0
        k2 = int(kk[0]) if len(kk) >= 2 else 0

        self.runner.enqueue(
            do_signal='Bill_do_StepForward',
            done_signal='Bill_StepForward_Done',
            ints={'Bill_StepForward_K': k1},
            timeout=30.0,
            # strings={'Bill_StepForward_StepSize': str(dist)},  # + random
            strings={'Bill_StepForward_StepSize': str(dist)},
            phase_id=0,
            target_id=p1
        )
        if ang == 0:
            pass
        else:
            self.runner.enqueue(
                do_signal='Bill_do_Rotate',
                done_signal='Bill_Rotate_Done',
                ints={'Bill_Rotate_K': k1, 'Bill_Rotate_Angle': - ang},
                phase_id=1,
                target_id=p1
            )

        self.runner.enqueue(
            do_signal='Bill_do_MoveHand',
            done_signal='Bill_MoveHand_Done',
            timeout=50.0,
            ints={'Bill_MoveHand_Dummy': self.places[p1-1], 'Bill_MoveHand_K': k1},
            phase_id=2,
            target_id=p1
        )
        if p2 == 0:
            return True

        self.runner.enqueue(
            do_signal='Bill_do_MoveHand',
            done_signal='Bill_MoveHand_Done',
            timeout=50.0,
            ints={'Bill_MoveHand_Dummy': self.place_init, 'Bill_MoveHand_K': k2},
            phase_id=3,
            target_id=p2
        )
        self.runner.enqueue(
            do_signal='Bill_do_StepSideways',
            done_signal='Bill_StepSideways_Done',
            timeout=30.0,
            ints={'Bill_StepSideways_K': k2},
            strings={'Bill_StepSideways_StepSize': str(0.23*(p1-p2))},  # + random
            phase_id=4,
            target_id=p2,
        )
        self.runner.enqueue(
            do_signal='Bill_do_MoveHand',
            done_signal='Bill_MoveHand_Done',
            timeout=50.0,
            ints={'Bill_MoveHand_Dummy': self.places[p2-1], 'Bill_MoveHand_K': k2},
            phase_id=2,
            target_id=p2
        )

    def move_robot(self, action):
        for j, a in zip(self.joints, action):
            self.sim.setJointTargetVelocity(j, a)

    def get_object_position(self, object):
        return self.sim.getObjectPosition(object)

    def get_object_orientation(self, object):
        return self.sim.getObjectOrientation(object)

    def get_joint_positions(self):
        return [self.sim.getJointPosition(h) for h in self.joints]

    def get_joint_velocities(self):
        return [self.sim.getJointVelocity(h) for h in self.joints]

    def stop_joint_velocities(self):
        return [self.sim.setJointTargetVelocity(h, 0) for h in self.joints]

    # vrep.py
    def clear_runner_queue(self):
        try:
            while self.runner.is_busy():
                self.runner.tick()
                self.sim.step()
        except Exception:
            pass
        try:
            self.runner.queue.clear()
        except Exception:
            self.runner.queue = []

    # def _py(self, obj):
    #     if isinstance(obj, (np.floating, np.integer, np.bool_)):
    #         return obj.item()
    #     if isinstance(obj, np.ndarray):
    #         # 若 API 需要 list，就转成 list；需要 bytes 就另外处理
    #         return obj.astype(float).tolist()
    #     if isinstance(obj, (list, tuple)):
    #         return [_py(x) for x in obj]
    #     return obj

    def _unit(self, v, eps=1e-12):
        v = np.asarray(v, dtype=float)
        n = np.linalg.norm(v)
        if n < eps:
            return v, 0.0
        return v / n, n

    def cylinder_normal_at_point(self, hole_xyz, center_xyz, axis_dir=(0., 1., 0.)):
        p = np.asarray(hole_xyz, dtype=float)
        c = np.asarray(center_xyz, dtype=float)
        a, _ = self._unit(axis_dir)

        # Projection onto a plane perpendicular to a：radial = (I - a a^T) (p - c)
        rel = p - c
        radial = rel - np.dot(rel, a) * a
        n_hat, nrm = self._unit(radial)
        if nrm == 0.0:
            raise ValueError("The hole point is on the axis and the normal is undefined")
        return n_hat

    def angle_deg(self, u, v):
        u, _ = self._unit(u)
        v, _ = self._unit(v)
        dot = np.clip(np.dot(u, v), -1.0, 1.0)
        return np.degrees(np.arccos(dot)), dot

    def ee_to_normal_angle(self, hole_xyz, fuselage_center_xyz, axis_dir=(0., 1., 0.), tcp_handle=None):
        n = self.cylinder_normal_at_point(hole_xyz, fuselage_center_xyz, axis_dir)
        if tcp_handle is not None:
            m = self.sim.getObjectMatrix(tcp_handle, -1)
            v = np.array([m[0], m[4], m[8]])
            v2 = np.array([m[1], m[5], m[9]])
            ee_vec, _ = self._unit(v)
            ee_vec2, _ = self._unit(v2)
            ang_ee, dot = self.angle_deg(ee_vec, n)
            wy = np.array([0., -1., 0.], dtype=float)
            ang_wy, _ = self.angle_deg(ee_vec2, wy)
        else:
            ang_ee = None; dot = None; ang_wy = None
        return n, ang_ee, dot, ang_wy

if __name__ == "__main__":
    v = VrepInterface(23000)
    # print("pos", v.get_object_position(v.Bill))
    # print("pos2", v.get_object_position(v.Bill)[:2])
    v.start()

    print('Bill handle =', v.Bill)
    print('Robot handle =', v.Robot)
    print('Targets =', v.targets)
    print('Joints  =', v.joints)

    q_now = v.get_joint_positions()
    print('q_now =', q_now)


    v.move_bill(v.places[2])
    for _ in range(100):
        if v.if_collision():
            break
        else:
            v.move_robot([0.05, 0, 0, 0, 0.01, 0.01])
            if v.runner.is_busy() or v.runner.queue:
                v.runner.tick()
                v.sim.step()
            else:
                v.sim.step()
    # print('velocity_now =', v.get_joint_velocities())
    v.stop_joint_velocities()
    v.reset_robot(q_now)
    v.reset_bill()

    time.sleep(3)
    v.stop()