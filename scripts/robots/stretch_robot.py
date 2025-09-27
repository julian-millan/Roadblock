import numpy as np
import warnings
import threading
import time

from .stretch_kinematics_solver import StretchKinematicsSolver
from . import BaseRobot
from vision.cameras import BaseCamera
from trajectories import Trajectory, Waypoint
import stretch_body.robot

# Gets the closest waypoint timestamp difference from Stretch API (this is in seconds).
from stretch_body.trajectories import WAYPOINT_ISCLOSE_ATOL

class StretchRobot(BaseRobot):
    def __init__(self, tool=None, stiffness=200):
        self.robot = stretch_body.robot.Robot()
        self.robot.startup()
        self.robot.base.reset_odometry()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if tool is None:
                self.kinematics_ee = StretchKinematicsSolver()
            else:
                self.kinematics_ee = StretchKinematicsSolver(goal_link=tool)
            self.kinematics_camera = StretchKinematicsSolver(goal_link="camera")
        self._joint_traj = []
        self.stiffness = stiffness
        self.exec_index = 0
        self._thread = None
        self._stop = None
        self._lock = threading.Lock()
    
    @property
    def is_trajectory_active(self) -> bool:
        t = self._thread
        return t is not None and t.is_alive()

    def pull_all_status(self):
        self.robot.wacc.pull_status()
        self.robot.base.pull_status()
        self.robot.lift.pull_status()
        self.robot.arm.pull_status()
        self.robot.pimu.pull_status()
        self.robot.head.pull_status()
        self.robot.end_of_arm.pull_status()

    def get_6dof_joints(self) -> np.ndarray:
        self.pull_all_status()
        status = self.robot.get_status()
        js = np.zeros(6)
        js[0] = status['base']['x']
        js[1] = status['lift']['pos']
        js[2] = status['arm']['pos']
        js[3] = status['end_of_arm']['wrist_yaw']['pos']
        js[4] = status['end_of_arm']['wrist_pitch']['pos']
        js[5] = status['end_of_arm']['wrist_roll']['pos']
        return np.copy(js)

    def get_head_joints(self) -> np.ndarray:
        self.pull_all_status()
        status = self.robot.get_status()
        js = np.zeros(2)
        js[0] = status['head']['head_pan']['pos']
        js[1] = status['head']['head_tilt']['pos']
        return np.copy(js)

    def get_camera_transform(self) -> np.ndarray:
        js = self.get_head_joints() 
        return self.kinematics_camera.forward_kinematics(js)
    
    def get_gripper_value(self) -> float:
        self.pull_all_status()
        status = self.robot.get_status()
        return status['end_of_arm']['stretch_gripper']['pos']

    def move_to_js(self, js):
        js_curr = self.get_6dof_joints()
        self.robot.base.translate_by(js[0]-js_curr[0])
        self.robot.lift.move_to(js[1])
        self.robot.arm.move_to(js[2])
        self.robot.end_of_arm.move_to('wrist_yaw', js[3])
        self.robot.end_of_arm.move_to('wrist_pitch', js[4])
        self.robot.end_of_arm.move_to('wrist_roll', js[5])
        self.robot.push_command()

    def append_cartesian_to_traj(self, dt, pos, orient, curr_js=None):
        if curr_js is None:
            if len(self._joint_traj) > 0:
                curr_js = self._joint_traj[-1]["js"]
            else:
                curr_js = self.get_6dof_joints()
        js = self.inverse_kinematics_orientation(pos, orient, js=curr_js)
        self.append_js_to_traj(dt, js)
        self._joint_traj[-1]["position"] = pos
        self._joint_traj[-1]["rotation"] = orient

    def append_js_to_traj(self, dt, js):
        next_time = (0 if len(self._joint_traj) == 0 else self._joint_traj[-1]["time"]) + dt
        self.robot.base.trajectory.add(time=next_time, x=js[0], y=0, theta=0)
        self.robot.lift.trajectory.add(t_s=next_time, x_m=js[1])
        self.robot.arm.trajectory.add(t_s=next_time, x_m=js[2])
        self.robot.end_of_arm.get_joint('wrist_yaw').trajectory.add(t_s=next_time, x_r=js[3])
        self.robot.end_of_arm.get_joint('wrist_pitch').trajectory.add(t_s=next_time, x_r=js[4])
        self.robot.end_of_arm.get_joint('wrist_roll').trajectory.add(t_s=next_time, x_r=js[5])
        self._joint_traj.append({"time": next_time, "js": js})

    def plan_low_level_trajectory(self, traj, camera = None):
        return super().plan_low_level_trajectory(traj, WAYPOINT_ISCLOSE_ATOL, camera)

    def _execute_trajectory(self, stop_event: threading.Event, low_level_traj: Trajectory, camera: BaseCamera | None = None, buffer_time = 5):
        def get_next_index(start_time: float, start_index=0) -> int:
            for i in range(start_index, len(self._joint_traj)):
                if self._joint_traj[i]["time"] >= time.perf_counter() - start_time:
                    return i
            return len(self._joint_traj)
    
        start_time = time.perf_counter()
        self.robot.base.trajectory.clear()
        self.robot.lift.trajectory.clear()
        self.robot.arm.trajectory.clear()
        self.robot.end_of_arm.get_joint('wrist_yaw').trajectory.clear()
        self.robot.end_of_arm.get_joint('wrist_pitch').trajectory.clear()
        self.robot.end_of_arm.get_joint('wrist_roll').trajectory.clear()
        self._joint_traj.clear()

        if camera:
            camera_transform = self.get_camera_transform()
        success = False
        self.input_index = 0
        self.exec_index = 0
        try:
            while not stop_event.is_set() and self.input_index < len(low_level_traj):
                # If motion has started and we have enough waypoints in the trajectory, then wait until the next waypoint is reached.
                if success:
                    self.exec_index = get_next_index(self.motion_start_time, start_index=self.exec_index)
                    if self._joint_traj[-1]["time"] - self._joint_traj[self.exec_index]["time"] >= buffer_time:
                        time.sleep(0.01)
                        continue
                # Manually add the current joint state to the trajectory at time=0, as this is required for Stretch.
                waypoint = low_level_traj[self.input_index]
                if self.input_index == 0:
                    curr_js = self.get_6dof_joints()
                    self.append_js_to_traj(0, curr_js)
                    fk = self.forward_kinematics(curr_js)
                    self._joint_traj[-1]["waypoint"] = Waypoint(position=fk[:3,3], velocity=waypoint.velocity, force=waypoint.force, gripper=self.get_gripper_value())
                    self._joint_traj[-1]["waypoint"].high_level_index = waypoint.high_level_index
                    self._joint_traj[-1]["position"] = fk[:3,3]
                    self._joint_traj[-1]["traj_index"] = 0
                    if waypoint.velocity < 1e-5:
                        waypoint.velocity = low_level_traj[1].velocity
                if waypoint.pause is not None and waypoint.pause > 0:
                    self.append_js_to_traj(waypoint.pause, self._joint_traj[-1]["js"])
                    self._joint_traj[-1]["waypoint"] = waypoint
                    self._joint_traj[-1]["position"] = self._joint_traj[-2]["position"]
                    self._joint_traj[-1]["traj_index"] = self.input_index
                    self.input_index += 1
                    continue
                # If force is provided, then compute the force-scaled displacements and hence positions of the end-effector.
                # We project the position to pixel space and back to 3D to ensure proper alignment with the surface.
                if waypoint.force != []:
                    assert camera is not None, "If force is provided, camera must also be provided."
                    projected_pixel = camera.get_pixel_from_3d(waypoint.position, camera_transform)
                    backprojected_position = camera.get_3d_from_pixel(projected_pixel, camera_transform)
                    backprojected_normal = camera.get_normal_from_pixel(projected_pixel, camera_transform)
                    position = backprojected_position - (waypoint.force[1] / self.stiffness) * backprojected_normal
                # If there is no force, then just use the position directly.
                else:
                    position = waypoint.position
                dt = np.linalg.norm(self._joint_traj[-1]["position"] - position) / waypoint.velocity
                # Skipping waypoints if too close together, but for the last waypoint, always add it, even at a lower speed.
                if dt <= WAYPOINT_ISCLOSE_ATOL:
                    if self.input_index == len(low_level_traj) - 1:
                        dt = WAYPOINT_ISCLOSE_ATOL + 0.01
                    else:
                        self.input_index += 1
                        continue
                self.append_cartesian_to_traj(dt, position, waypoint.rotation)
                self._joint_traj[-1]["waypoint"] = waypoint
                self._joint_traj[-1]["traj_index"] = self.input_index
                # Starts trajectory after the first waypoints are added, and keep adding waypoints afterwards.
                if self._joint_traj[-1]["time"] >= buffer_time and not success:
                    success = self.robot.follow_trajectory(move_to_start_point=True)
                    if success:
                        self.motion_start_time = time.perf_counter()
                        print(f"Motion started in {self.motion_start_time - start_time:.3f}s.")
                self.input_index += 1
        finally:
            if stop_event.is_set():
                self.robot.stop_trajectory()
            else:
                # If motion never started (could be input trajectory shorter than buffer), then try a final attempt to start, if possible.
                if not success and self.input_index >= len(low_level_traj):
                    if len(self._joint_traj) > 0:
                        success = self.robot.follow_trajectory(move_to_start_point=True)
                # Ensure trajectory fully executed before exiting.
                while success and self.exec_index < len(self._joint_traj):
                    self.exec_index = get_next_index(self.motion_start_time, start_index=self.exec_index)
                    time.sleep(0.01)


    def execute_trajectory(self, low_level_traj: Trajectory, camera: BaseCamera | None = None):
        with self._lock:
            # Signal previous worker (if any) to stop and wait for it
            if self._thread and self._thread.is_alive():
                self._stop.set()
                self._thread.join(timeout=None)
                if self._thread.is_alive():
                    # Still running; you can either wait longer or log and continue.
                    print("Warning: previous worker didn't stop in time; starting a new one anyway.")

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._execute_trajectory, args=(self._stop, low_level_traj, camera, 10))
        self._thread.daemon = True
        self._thread.start()
        time.sleep(1.0)  # Give the thread a moment to start

    def stop(self, timeout=None):
        with self._lock:
            if self._thread and self._thread.is_alive():
                self._stop.set()
                self._thread.join(timeout=timeout)

    def forward_kinematics(self, js: None | np.ndarray = None) -> np.ndarray:
        if js is None:
            js = self.get_6dof_joints()
        return self.kinematics_ee.forward_kinematics(js)

    def inverse_kinematics(self, pos: np.ndarray, js: None | np.ndarray = None) -> np.ndarray:
        if js is None:
            js = self.get_6dof_joints()
        return self.kinematics_ee.inverse_kinematics(pos, js)
        
    def inverse_kinematics_orientation(self, pos: np.ndarray, orientation: np.ndarray, js: None | np.ndarray = None) -> np.ndarray:
        if js is None:
            js = self.get_6dof_joints()
        transform = np.eye(4)
        transform[0:3,0:3] = orientation
        transform[0:3,3] = pos
        return self.kinematics_ee.inverse_kinematics_orientation(transform, js)
