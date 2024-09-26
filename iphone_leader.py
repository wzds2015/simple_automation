import enum
import logging
import math
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import tqdm
import mink
from mujoco_ar import MujocoARConnector
import mujoco
import mujoco.viewer
import torch


from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc

class IPhoneLeader:

    def __init__(
        self,
        configuration,
        frame_name,
        frame_type,
        robot_name,
    ):
        self.configuration = configuration
        self.frame_name = frame_name
        self.frame_type = frame_type
        self.robot_name = robot_name
        self.motors_pos = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
        self.is_connected = False
        self.mujocoAR = None
        print("constructor")
        pass

        
    def connect(self):
        self.is_connected = True

        # init_pos_rad = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0])
        # self.motors_pos = init_pos_rad

        model = self.configuration.model
        data = self.configuration.data

        collision_pairs = [
            # (["wrist_3_link"], ["floor", "wall", *[f"green_box{i}" for i in range(1, 10)]]),
        ]

        self.limits = [
            mink.ConfigurationLimit(model=model),
            mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs),
        ]
        if self.robot_name == "ur5e":
            max_velocities = {
                "shoulder_pan": np.pi,
                "shoulder_lift": np.pi,
                "elbow": np.pi,
                "wrist_1": np.pi,
                "wrist_2": np.pi,
                "wrist_3": np.pi,
            }        
        elif self.robot_name == "lerobot":
            max_velocities = {
                "elbow_flex_joint": np.pi, 
                "gripper_joint": np.pi, 
                "shoulder_lift_joint": np.pi, 
                "shoulder_pan_joint": np.pi, 
                "wrist_flex_joint": np.pi, 
                "wrist_roll_joint": np.pi,
            }
        elif self.robot_name == "lc":
            max_velocities = {
                "joint1": np.pi, 
                "joint2": np.pi, 
                "joint3": np.pi, 
                "joint4": np.pi, 
                "joint5": np.pi, 
            }
        else:
            raise KeyError(f"Robot name not found supported: {self.robot_name}")
        velocity_limit = mink.VelocityLimit(model, max_velocities)
        self.limits.append(velocity_limit)

        self.end_effector_task = mink.FrameTask(
                frame_name=self.frame_name,
                frame_type=self.frame_type,
                position_cost=1.0,
                orientation_cost=1.0,
                lm_damping=1.0,
            )

        self.tasks = [self.end_effector_task ]  


        self.configuration.update_from_keyframe("home")
        # mink.move_mocap_to_frame(model, data, "target", self.frame_name, self.frame_type)

        self.mujocoAR = MujocoARConnector(mujoco_model=model, mujoco_data=data, port=8888)
        self.mujocoAR.start()

        initial_pos = data.body("target").xpos.copy()
        initial_mat = data.body("target").xmat.copy().reshape((3,3))
    
        self.mujocoAR.link_body(
            name="target",
            scale=2.0,
            position_origin=initial_pos,
            rotation_origin=initial_mat,
        )


        # self.write("Initial Pos", values=initialPosNd)

        pass

    def _readInRadians(self):

        values = []

        for i in range(6):
            values.append(self.motors_pos[i])

        return np.asarray(values)
    
    def read(self, data_name, motor_names: str | list[str] | None = None):

        configuration = self.configuration
        model = configuration.model
        data = configuration.data
        limits = self.limits
        tasks = self.tasks

        T_wt = mink.SE3.from_mocap_name(model, data, "target")
        self.end_effector_task.set_target(T_wt)

        dt_s = 1 / 30.0

        vel = mink.solve_ik(
            configuration, tasks, dt_s, "quadprog", 1e-3, limits=limits
        )

        d = vel * 0.01
        new_position_delta = d[-6:]

        new_position_rad = self._readInRadians() + new_position_delta
        self.motors_pos = new_position_rad
        new_position_degrees = np.rad2deg(new_position_rad)
        res = new_position_degrees * 100
        res = torch.from_numpy(res)
        res = res.numpy().astype(np.int32)
        return res

    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):
        pass

    def set_calibration(self, calibration: dict[str, tuple[int, bool]]):
        self.calibration = calibration

    def disconnect(self):
        if(self.is_connected):
            if(self.mujocoAR):
                self.mujocoAR.stop()
            self.is_connected = False

    def __del__(self):
        # if getattr(self, "is_connected", False):
        self.disconnect()

if __name__ == "__main__":
    # Helper to find the usb port associated to all your DynamixelMotorsBus.
    pass
