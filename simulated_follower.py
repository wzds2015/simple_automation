import enum
import logging
import math
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import tqdm
import mujoco


from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc

class SimulatedFollower:

    def __init__(
        self,
        configuration,
    ):
        self.configuration = configuration
        self.old_pos = np.zeros(12)
        pass

    def connect(self):
        self.is_connected = True
        self.data = self.configuration.data
        self.model = self.configuration.model

        init_pos_rad = [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]
        self.data.qpos[-6:] = init_pos_rad
        self.old_pos = deepcopy(self.data.qpos[-6:])
        # deep copy
        mujoco.mj_forward(self.model, self.data)

        pass

    def read(self, data_name, motor_names: str | list[str] | None = None):
        old_pos = self.old_pos[-6:]        
        pos_in_deg = np.rad2deg(old_pos)
        res = pos_in_deg * 100
        return res

    def set_calibration(self, calibration: dict[str, tuple[int, bool]]):
        self.calibration = calibration

    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):

        if data_name in ["Torque_Enable", "Operating_Mode", "Homing_Offset", "Drive_Mode", "Position_P_Gain", "Position_I_Gain", "Position_D_Gain"]:
            return np.array(None)

        self.old_pos = deepcopy(self.data.qpos[-6:])

        pos_in_degrees = values / 100

        # pos_in_degrees = values
        pos_in_rad = np.deg2rad(pos_in_degrees)


        self.data.qpos[-6:] = pos_in_rad
        # old

        # viewer.sync()
        # old_pos = self.data.qpos[-6:]
        # vel = (pos_in_rad - old_pos) * 500.0

        # dt_s = 1/30.0
        # self.configuration.integrate_inplace(vel, dt_s)

        pass

    def disconnect(self):
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

if __name__ == "__main__":
    pass
