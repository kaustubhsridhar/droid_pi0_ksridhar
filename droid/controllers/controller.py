from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from droid.controllers.robot_utils import Proprio, position_action_to_delta_action

@dataclass
class ControllerConfig:
    server_ip: str = "tcp://172.16.0.4:4242"
    max_pos_delta: float = 0.05
    max_euler_delta: float = 0.2


class Controller:
    def __init__(self, cfg: ControllerConfig):
        # Remove ZeroRPC connection entirely
        print("Initializing local droid controller")
        
        # Import your actual robot control library here
        # import your_robot_library as robot_lib
        # self.robot = robot_lib.Robot()  # or whatever initialization is needed
        
        self.home = np.pi * np.array([0, -0.25, 0, -0.75, 0, 0.5, 0], dtype=np.float32)
        self.cfg = cfg
        self.action_dim = 7
        self.curr_proprio = None

    def reset(self, randomize: bool) -> None:
        print(f"[controller] randomize? {randomize}")
        # Implement reset using your actual robot control library
        # self.robot.reset(randomize)

    def get_proprio(self) -> Proprio:
        # Implement using your actual robot's state retrieval method
        # robot_state = self.robot.get_state()  # or whatever the actual method is
        
        # For testing/development, you can return mock data:
        mock_pos = np.array([0.5, 0.0, 0.5])
        mock_quat = np.array([0.0, 0.0, 0.0, 1.0])  # w,x,y,z
        
        proprio = Proprio(
            eef_pos=mock_pos,
            eef_quat=mock_quat,
            gripper_open=1.0,
        )
        return proprio

    def delta_control(self, delta_pos: np.ndarray, delta_euler: np.ndarray, gripper_open: float):
        # (optional?) clip with max delta
        curr_proprio = self.get_proprio()
        delta_pos = np.clip(delta_pos, -self.cfg.max_pos_delta, self.cfg.max_pos_delta)
        delta_euler = np.clip(delta_euler, -self.cfg.max_euler_delta, self.cfg.max_euler_delta)

        # compute new pos and new quat
        new_pos = curr_proprio.eef_pos + delta_pos
        curr_rot = Rotation.from_euler("xyz", curr_proprio.eef_euler)
        delta_rot = Rotation.from_euler("xyz", delta_euler)
        new_quat = (delta_rot * curr_rot).as_quat()

        # Implement control using your actual robot control library
        # self.robot.move_to_pose(new_pos, new_quat, gripper_open)
        print(f"Delta control: pos={delta_pos}, euler={delta_euler}, gripper={gripper_open}")

    def position_control(self, new_pos: np.ndarray, new_euler: np.ndarray, gripper_open: float):
        new_quat = Rotation.from_euler("xyz", new_euler).as_quat()
        # Implement control using your actual robot control library
        # self.robot.move_to_pose(new_pos, new_quat, gripper_open)
        print(f"Position control: pos={new_pos}, euler={new_euler}, gripper={gripper_open}")

    def position_to_delta(self, new_pos: np.ndarray, new_euler: np.ndarray):
        proprio = self.get_proprio()
        return position_action_to_delta_action(
            proprio.eef_pos, proprio.eef_euler, new_pos, new_euler
        )
