from abc import ABC, abstractmethod
import numpy as np

from vision.cameras import BaseCamera
from trajectories import Trajectory, Waypoint

class BaseRobot(ABC):
    """ Base class for robot wrappers.
    If implementing your own robot wrapper, you should inherit from this class and implement the following methods:
    - get_camera_transform
    - get_gripper_value
    - plan_low_level_trajectory
    - execute_trajectory
    - forward_kinematics
    - get_next_waypoint_index
    - get_speech_index
    - stop
    """

    @property
    @abstractmethod
    def is_trajectory_active(self) -> bool:
        pass

    @abstractmethod
    def get_camera_transform(self) -> np.ndarray:
        """
        Returns the transformation matrix (4x4) from camera frame to robot base frame.
        This can be either a fixed transform or a dynamic one (e.g. if the camera connected to the robot base through known joints).
        """
        pass

    @abstractmethod
    def get_gripper_value(self) -> float:
        """
        Returns the current gripper value. We assume <0 is closed, >0 is open.
        """
        pass

    @abstractmethod
    def plan_low_level_trajectory(self, traj: Trajectory, camera: None | BaseCamera = None) -> Trajectory:
        """
        Given a high-level trajectory (sequence of waypoints, assuming linear interpolation between waypoints), plans a low-level trajectory that's denser
        and suitable for smooth execution on the robot. 
        Args:
            traj (Trajectory): High-level trajectory to be converted to low-level trajectory.
            camera (BaseCamera, optional): Camera object to be used for determining surface normals. Required if any waypoint involves force.
        Returns:
            low_level_traj (Trajectory): Low-level trajectory suitable for execution on the robot.
        """
        pass
    
    @abstractmethod
    def execute_trajectory(self, low_level_traj: Trajectory, **kwargs):
        """
        Executes a low-level trajectory on the robot. This should be non-blocking.
        Args:
            low_level_traj (Trajectory): Low-level trajectory to be executed on the robot.
        """
        pass

    @abstractmethod
    def stop(self):
        """
        Stops any active trajectory.
        """
        pass

    @abstractmethod
    def forward_kinematics(self, js: None | np.ndarray = None) -> np.ndarray:
        """
        Computes the forward kinematics (4x4) for the robot's end-effector. If js is None, uses the current joint states.
        Args:
            js (np.ndarray, optional): Joint states to be used for computing forward kinematics.
        Returns:
            transform (np.ndarray): Transformation matrix (4x4) representing the end-effector pose.
        """
        pass

    def plan_low_level_trajectory(self, traj: Trajectory, time_spacing: float, camera: None | BaseCamera = None) -> Trajectory:
        self._low_level_waypoint_traj = Trajectory()
        for i in range(len(traj)):
            # For the first waypoint, make sure that it is close to the current end-effector position, and don't consider any motion.
            if i == 0:
                fk = self.forward_kinematics()
                assert(np.linalg.norm(traj[i].position - fk[:3,3]) < 0.01)
                traj[i].rotation = fk[:3,:3]
                self._low_level_waypoint_traj.add_waypoint(traj[i])
                self._low_level_waypoint_traj[-1].high_level_index = i
                continue
            # If there is no rotation provided for the waypoint, then use the last rotation in the trajectory.
            if traj[i].rotation is None:
                traj[i].rotation = self._low_level_waypoint_traj[-1].rotation
            # If pause is provided, then assume the position and orientation are the same as the previous waypoint.
            if traj[i].pause is not None and traj[i].pause > 0:
                pre_pause_waypoint = traj[i].copy()
                pre_pause_waypoint.pause = 0
                self._low_level_waypoint_traj.add_waypoint(pre_pause_waypoint)
                self._low_level_waypoint_traj[-1].high_level_index = i
                self._low_level_waypoint_traj.add_waypoint(traj[i])
                self._low_level_waypoint_traj[-1].high_level_index = i
                continue
            # Determine the number of samples to take between the two waypoints. Assumes the provided atol is the minimum, so give a little bit of margin.
            num = int(np.linalg.norm(traj[i].position - traj[i-1].position) / traj[i].velocity / (time_spacing / 0.9))
            if num < 1:
                raise RuntimeError(f"The waypoints indexed {i-1} and {i} are too close together, or the velocity is too high. Please increase the distance or decrease the velocity.")
            # If force is provided, then determine where the surface is to apply the force to.
            if traj[i].force != []:
                assert camera is not None, "If force is provided, camera must also be provided."
                assert traj[i].start_pixel is not None and traj[i].end_pixel is not None, "If force is provided, start_pixel and end_pixel are required to properly compute a low-level trajectory with contact."
                camera_transform = self.get_camera_transform()
                num = min(num, abs(traj[i].end_pixel[0] - traj[i].start_pixel[0]), abs(traj[i].end_pixel[1] - traj[i].start_pixel[1]))
                if len(traj[i].force) == 1:
                    forces = np.array([traj[i].force[0]] * num)
                elif len(traj[i].force) == 2:
                    forces = np.linspace(traj[i].force[0], traj[i].force[1], num, endpoint=True)
                # Interpolate pixels, get surface points and normals.
                pixels = np.linspace(traj[i].start_pixel, traj[i].end_pixel, num, endpoint=True, dtype=np.uint16)
                points = np.array([camera.get_3d_from_pixel(pixel=pixel, transform=camera_transform) for pixel in pixels])
                normals = np.array([camera.get_normal_from_pixel(pixel=pixel, transform=camera_transform) for pixel in pixels])
                # Compute the mean normal and perform smoothening of the normals based on the mean.
                mean_normal = np.mean(normals, axis=0)
                mean_normal /= np.linalg.norm(mean_normal)
                smoothened_normals = 0.9 * mean_normal + 0.1 * normals
                smoothened_normals /= np.linalg.norm(smoothened_normals, axis=1)[:, np.newaxis]
                # Compute the orientations if surface alignment function is provided, or just use a fixed orientation.
                orientations = [traj[i].surface_align_fn(norm) if traj[i].surface_align_fn is not None else traj[i].rotation for norm in smoothened_normals]
                assert np.array(orientations[0]).shape == (3,3), "surface_align_fn must return a 3x3 rotation matrix."
                for j in range(num):
                    if i == 1 and j == 0:
                        continue
                    self._low_level_waypoint_traj.add_waypoint(Waypoint(position=points[j], velocity=(traj[i-1].velocity if j == 0 else traj[i].velocity), force=([forces[j],forces[j]] if j == 0 else [forces[j-1], forces[j]]), gripper=traj[i].gripper, rotation=orientations[j], surface_normal=smoothened_normals[j], start_pixel=pixels[j-1], end_pixel=pixels[j]))
                    self._low_level_waypoint_traj[-1].high_level_index = i-1 if j == 0 else i
                    if hasattr(traj[i-1], 'no_position_update'):
                        self._low_level_waypoint_traj[-1].no_position_update = traj[i-1].no_position_update
            # If there is no force, then just interpolate the positions and orientations linearly.
            else:
                num = min(num,2) # don't really need to interpolate because major joints on Stretch are all prismatic
                positions = np.linspace(traj[i-1].position, traj[i].position, num, endpoint=False)
                orientations = np.linspace(traj[i-1].rotation, traj[i].rotation, num, endpoint=False)
                for j in range(num):
                    if i == 1 and j == 0:
                        continue
                    self._low_level_waypoint_traj.add_waypoint(Waypoint(position=positions[j], velocity=(traj[i-1].velocity if j == 0 else traj[i].velocity), force=[], gripper=traj[i].gripper, rotation=orientations[j]))
                    self._low_level_waypoint_traj[-1].high_level_index = i-1 if j == 0 else i
                    if hasattr(traj[i-1], 'no_position_update'):
                        self._low_level_waypoint_traj[-1].no_position_update = traj[i-1].no_position_update
            if i == len(traj)-1:
                self._low_level_waypoint_traj.add_waypoint(traj[i])
                self._low_level_waypoint_traj[-1].high_level_index = i
                self._low_level_waypoint_traj[-1].no_position_update = True
        return self._low_level_waypoint_traj

    def convert_camera_to_base(self, camera_point: np.ndarray, transform_camera_base: None | np.ndarray = None, rotation_only=False) -> np.ndarray:
        if transform_camera_base is None:
            transform_camera_base = self.get_camera_transform()
        rotated_point = transform_camera_base[0:3,0:3] @ camera_point
        if rotation_only:
            return rotated_point
        else:
            return rotated_point + transform_camera_base[0:3,3]
        
    def convert_base_to_camera(self, base_point: np.ndarray, transform_camera_base: None | np.ndarray = None, rotation_only=False) -> np.ndarray:
        if transform_camera_base is None:
            transform_camera_base = self.get_camera_transform()
        rotated_point = transform_camera_base[0:3,0:3].T @ base_point
        if rotation_only:
            return rotated_point
        else:
            return rotated_point - transform_camera_base[0:3,0:3].T @ transform_camera_base[0:3,3]