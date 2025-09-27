import numpy as np
from scipy.spatial.transform import Rotation
import threading
import time
from xarm.wrapper import XArmAPI

from .base_robot import BaseRobot

class XArmRobot(BaseRobot):
    def __init__(self, ip, camera_transform, tcp_transform=np.eye(4)):
        self.robot = XArmAPI(ip, is_radian=True)
        self.robot.motion_enable(enable=True)
        self.robot.set_state(0)
        self.robot.set_mode(1)
        self.robot.set_state(0)
        self.robot.set_gripper_mode(0)
        self.robot.set_gripper_enable(True)

        self.camera_transform = np.array(camera_transform)
        self.tcp_transform = np.array(tcp_transform)

        self._thread = None
        self._stop = None
        self._lock = threading.Lock()
    
    @property
    def is_trajectory_active(self):
        t = self._thread
        return t is not None and t.is_alive()
    
    def get_camera_transform(self):
        return self.camera_transform

    def get_gripper_value(self):
        code, value = self.robot.get_gripper_position()
        if code != 0:
            print("Failed to get xArm gripper value.")
            return
        value -= 400
        return value

    def plan_low_level_trajectory(self, traj, camera = None):
        time_spacing = 1.0
        return super().plan_low_level_trajectory(traj, time_spacing, camera)
    
    def _execute_trajectory(self, stop_event, low_level_traj):
        start_time = time.perf_counter()
        curr_fk = self.forward_kinematics()
        try:
            for i in range(len(low_level_traj)):
                curr_seg_start = time.perf_counter()
                curr_seg_end = start_time + low_level_traj[i].timestamp
                next_fk = np.eye(4)
                next_fk[:3,3] = low_level_traj[i].position
                next_fk[:3,:3] = low_level_traj[i].rotation
                speed = np.linalg.norm(next_fk[:3,3] - curr_fk[:3,3]) / (curr_seg_end - curr_seg_start)
                while not stop_event.is_set() and time.perf_counter()-start_time < low_level_traj[i].timestamp:
                    time_in_segment = time.perf_counter() - curr_seg_start
                    interpolated_fk = (1 - time_in_segment / (curr_seg_end - curr_seg_start)) * curr_fk + \
                                    (time_in_segment / (curr_seg_end - curr_seg_start)) * next_fk
                    interpolated_fk = interpolated_fk @ np.linalg.inv(self.tcp_transform)
                    pose = self.transform_to_pose(interpolated_fk)
                    self.robot.set_servo_cartesian(pose, speed=speed)
                    time.sleep(0.01)
                curr_fk = next_fk
        finally:
            if stop_event.is_set():
                self.robot.stop_trajectory()
                
    def execute_trajectory(self, low_level_traj, camera = None):
        with self._lock:
            # Signal previous worker (if any) to stop and wait for it
            if self._thread and self._thread.is_alive():
                self._stop.set()
                self._thread.join(timeout=None)
                if self._thread.is_alive():
                    # Still running; you can either wait longer or log and continue.
                    print("Warning: previous worker didn't stop in time; starting a new one anyway.")

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._execute_trajectory, args=(self._stop, low_level_traj,))
        self._thread.daemon = True
        self._thread.start()
        return self._thread

    def transform_to_pose(self, transform):
        pose = np.zeros(6)
        pose[:3] = transform[:3,3] * 1e3
        pose[3:] = Rotation.from_matrix(transform[:3,:3]).as_euler("xyz")
        return pose
    
    def pose_to_transform(self, pose):
        transform = np.eye(4)
        transform[:3,3] = pose[:3] / 1e3
        transform[:3,:3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
        return transform

    def forward_kinematics(self, js=None):
        if js is None:
            code, pose = self.robot.get_position()
        else:
            code, pose = self.robot.get_forward_kinematics(js)
        if code != 0:
            print("Failed to get xArm Cartesian position.")
            return
        transform = self.pose_to_transform(np.array(pose))
        transform = transform @ self.tcp_transform
        return transform

    def stop(self):
        self.robot.motion_enable(enable=False)
        self.robot.disconnect()
