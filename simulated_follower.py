import enum
import logging
import math
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np
import tqdm


from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.utils.utils import capture_timestamp_utc

class SimulatedFollower:

    def __init__(
        self,
        configuration,
    ):
        self.qpos = np.zeros(12)
        self.configuration = configuration
        pass

    def connect(self):
        self.is_connected = True
        self.data = self.configuration.data
        self.model = self.configuration.model
        pass

    def read(self, data_name, motor_names: str | list[str] | None = None):
        old_pos = self.data.qpos[-6:]        
        return old_pos

    def set_calibration(self, calibration: dict[str, tuple[int, bool]]):
        self.calibration = calibration

    def write(self, data_name, values: int | float | np.ndarray, motor_names: str | list[str] | None = None):

        if data_name in ["Torque_Enable", "Operating_Mode", "Homing_Offset", "Drive_Mode", "Position_P_Gain", "Position_I_Gain", "Position_D_Gain"]:
            return np.array(None)

        pos_in_degrees = values / 1000.0

        # pos_in_degrees = values
        pos_in_rad = np.deg2rad(pos_in_degrees)

        # вариант с позицией новой
        # self.data.qpos[-6:] = pos_in_rad

        old_pos = self.data.qpos[-6:]
        vel = (pos_in_rad - old_pos) * 500.0

        dt_s = 0.002
        self.configuration.integrate_inplace(vel, dt_s)

        pass

    def disconnect(self):
        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()

if __name__ == "__main__":
    pass
