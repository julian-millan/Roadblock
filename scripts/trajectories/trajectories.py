import copy
import numpy as np
from typing import Iterator, overload

class Waypoint:
    """ A class representing a waypoint in a robot trajectory.
    Attributes:
        position (numpy.ndarray): The position of the waypoint in 3D Cartesian coordinates, relative to the robot's base frame.
        velocity (float): The desired velocity magnitude from the previous waypoint to the current waypoint.
        force (list[float]): The desired force magnitude profile from the previous waypoint to the current waypoint. Length <=2. Empty list means no contact, and list of zeros means contact, but no (noticeable) force.
        gripper (float): The desired gripper value at the waypoint, where a negative value is closed and a positive value is open.
        rotation (numpy.ndarray, optional): The desired rotation matrix at the waypoint. If not provided, it is assumed to stay the same as the last rotation in the trajectory.
        joint_state (arraylike, optional): The joint state of the robot at the waypoint. If not provided, then could be computed through inverse kinematics.
        pause (float, optional): The amount of pause from the previous waypoint to the current waypoint, in seconds. Providing this will assume the position and rotation are the same as the previous waypoint.
        surface_align_fn (callable, optional): A function that takes a surface normal and returns a rotation matrix to align the robot's end-effector to. If provided, the rotation field will be ignored. Only applies when there is force.
        start_pixel (numpy.ndarray, optional): The starting pixel in the camera image for surface alignment.
        end_pixel (numpy.ndarray, optional): The ending pixel in the camera image for surface alignment.
        surface_normal (numpy.ndarray, optional): When there is contact with a surface, it might be useful to also include the surface normal.
    """
    def __init__(self, position,  velocity, force, gripper, rotation=None, joint_state=None, pause=0.0, surface_align_fn=None, start_pixel=None, end_pixel=None, surface_normal=None):
        self.position = position
        self.velocity = velocity
        assert velocity >= 0, "Velocity must be a nonnegative value."
        if isinstance(force, (int, float)):
            self.force = [force, force]
        elif isinstance(force, list):
            assert len(force) <= 2, "Force must be a list of less than length 2 when passed as a list."
            if len(force) == 1:
                self.force = [force[0], force[0]]
            else:
                self.force = force
        self.gripper = gripper
        self.rotation = rotation
        self.joint_state = joint_state
        self.pause = pause
        self.surface_align_fn = surface_align_fn            
        self.start_pixel = start_pixel
        self.end_pixel = end_pixel
        self.timestamp = None
        self.surface_normal = surface_normal

    def __str__(self):
        return f"Waypoint(timestamp={self.timestamp}, position={self.position}, velocity={self.velocity}, force={self.force}, gripper={self.gripper})"
    
    def __deepcopy__(self, memo) -> 'Waypoint':
        if id(self) in memo:
            return memo[id(self)]
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                setattr(result, key, np.copy(value))
            elif isinstance(value, list):
                setattr(result, key, value.copy())
            else:
                setattr(result, key, copy.deepcopy(value, memo))
        return result
    
    def copy(self) -> 'Waypoint':
        """ Returns a deep copy of the waypoint. """
        return copy.deepcopy(self)

class Trajectory:
    def __init__(self):
        self._traj = []

    def __len__(self):
        return len(self._traj)
    
    @overload
    def __getitem__(self, index: int) -> Waypoint: ...

    @overload
    def __getitem__(self, index: slice) -> 'Trajectory': ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            new_traj = Trajectory()
            new_traj._traj = self._traj[index]
            return new_traj
        else:
            return self._traj[index]

    def __setitem__(self, index, waypoint):
        self._traj[index] = waypoint

    def __delitem__(self, index):
        del self._traj[index]

    def __iter__(self) -> Iterator[Waypoint]:
        for waypoint in self._traj:
            yield waypoint

    def __str__(self):
        output_str = "Trajectory:\n"
        for i, waypoint in enumerate(self._traj):
            output_str += f"Waypoint {i+1}: {waypoint}\n"
        return output_str
    
    def __deepcopy__(self, memo) -> 'Trajectory':
        new_traj = Trajectory()
        new_traj._traj = [copy.deepcopy(waypoint, memo) for waypoint in self._traj]
        return new_traj

    def add_waypoint(self, waypoint: Waypoint):
        waypoint = copy.deepcopy(waypoint)
        if len(self._traj) == 0:
            waypoint.timestamp = 0.0
            self._traj.append(waypoint)
            return
        last_waypoint = self._traj[-1]
        if waypoint.pause > 0:
            dt = waypoint.pause
            waypoint.position = last_waypoint.position
            waypoint.rotation = last_waypoint.rotation
        else:
            dt = np.linalg.norm(last_waypoint.position - waypoint.position) / waypoint.velocity
        waypoint.timestamp = last_waypoint.timestamp + dt
        self._traj.append(waypoint)

    def copy(self) -> 'Trajectory':
        """ Returns a deep copy of the trajectory. """
        return copy.deepcopy(self)