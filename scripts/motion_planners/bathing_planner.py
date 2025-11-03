import numpy as np
from scipy.spatial.transform import Rotation

from robots import BaseRobot
from trajectories import Trajectory, Waypoint
from vision.cameras import BaseCamera
from vision.pose_detector import PoseDetector

class BathingPlanner:
    def __init__(self):
        pass

    def bathing_orientation_fn(self, motion_dir):
        def thunk(surface_normal):
            x_dir = np.cross(surface_normal, motion_dir)
            x_dir = x_dir / np.linalg.norm(x_dir)
            orientation = np.array([x_dir, np.cross(surface_normal, x_dir), surface_normal]).T
            orientation = orientation @ Rotation.from_euler('y', 40, degrees=True).as_matrix()
            return orientation
        return thunk

    def plan_trajectory(self, robot: BaseRobot, camera: BaseCamera, detector: PoseDetector, side="left", trajectory_num=1) -> list[Waypoint]:
        fk = robot.forward_kinematics()
        camera_transform = robot.get_camera_transform()
        gripper_value = robot.get_gripper_value()
        start_point = fk[0:3,3]
        start_orientation = fk[0:3,0:3]
        start_up = start_point + np.array([0,0,0.1])
        wrist_pixel = detector.query_landmark(side + " wrist")
        # elbow_pixel = detector.query_landmark(side + " elbow")
        # Gemini, the above code should remain the same. Add the roadblock pixels explicitly below.
        # Gemini, you need to get the elbow pixel from the image (dont query landmarks), then transform the roadblock pixels to points below
        wrist_point = camera.get_3d_from_pixel(pixel=wrist_pixel, transform=camera_transform)
        elbow_point = camera.get_3d_from_pixel(pixel=elbow_pixel, transform=camera_transform)
        

        # This next code block should remain unchanged
        forearm_dir = (elbow_point - wrist_point) / np.linalg.norm(elbow_point - wrist_point)
        path_pixels = np.linspace(wrist_pixel, elbow_pixel, 10, dtype=np.uint16, endpoint=True)
        path_normals = [camera.get_normal_from_pixel(pixel=pixel, transform=camera_transform) for pixel in path_pixels]
        bathe_start_orientation = self.bathing_orientation_fn(forearm_dir)(np.mean(path_normals, axis=0))
        

        if trajectory_num == 1:
            elbow_up = elbow_point + np.array([0,0,0.1])
            wrist_up = wrist_point + np.array([0,0,0.04])
            # Reason about potential roadblock_up points here, if needed

            trajectory = Trajectory()
            # Add trajectory waypoints to navigate any potential roadblocks, be sure to integrate it smoothly into the trajectory
            trajectory.add_waypoint(Waypoint(start_point, rotation=start_orientation, velocity=0, force=[], gripper=gripper_value))
            trajectory.add_waypoint(Waypoint(wrist_up, rotation=bathe_start_orientation, velocity=0.03, force=[], gripper=gripper_value))
            trajectory.add_waypoint(Waypoint(wrist_point, rotation=bathe_start_orientation, velocity=0.02, force=[], gripper=gripper_value))
            trajectory.add_waypoint(Waypoint(elbow_point, rotation=None, velocity=0.01, force=[1.0, 3.0], gripper=gripper_value, surface_align_fn=self.bathing_orientation_fn(forearm_dir), start_pixel=wrist_pixel, end_pixel=elbow_pixel))
            trajectory.add_waypoint(Waypoint(elbow_up, rotation=None, velocity=0.02, force=[], gripper=gripper_value))
            trajectory.add_waypoint(Waypoint(start_up, rotation=start_orientation, velocity=0.03, force=[], gripper=gripper_value))
            trajectory.add_waypoint(Waypoint(start_point, rotation=start_orientation, velocity=0.04, force=[], gripper=gripper_value))

        elif trajectory_num == 2:
            # Reason about potential roadblocks around the shoulder add repeat your pixel and point inclusions while including these new roadblocks
            shoulder_pixel = detector.query_landmark(side + " shoulder")
            shoulder_point = camera.get_3d_from_pixel(pixel=shoulder_pixel, transform=camera_transform)
            shoulder_normal = camera.get_normal_from_pixel(pixel=shoulder_pixel, transform=camera_transform)
            shoulder_up = shoulder_point + shoulder_normal * 0.1
            wrist_up = wrist_point + np.array([0,0,0.04])

            trajectory = Trajectory()
            # Add trajectory waypoints to navigate any potential roadblocks, be sure to integrate it smoothly into the trajectory
            trajectory.add_waypoint(Waypoint(start_point, rotation=start_orientation, velocity=0, force=[], gripper=gripper_value))
            trajectory.add_waypoint(Waypoint(wrist_up, rotation=bathe_start_orientation, velocity=0.03, force=[], gripper=gripper_value))
            trajectory.add_waypoint(Waypoint(wrist_point, rotation=bathe_start_orientation, velocity=0.03, force=[], gripper=gripper_value))
            trajectory.add_waypoint(Waypoint(elbow_point, rotation=None, velocity=0.01, force=[1.0, 1.0], gripper=gripper_value, surface_align_fn=self.bathing_orientation_fn(forearm_dir), start_pixel=wrist_pixel, end_pixel=elbow_pixel))
            trajectory.add_waypoint(Waypoint(shoulder_point, rotation=None, velocity=0.04, force=[1.0, 1.0], gripper=gripper_value, start_pixel=elbow_pixel, end_pixel=shoulder_pixel))
            trajectory.add_waypoint(Waypoint(shoulder_up, rotation=None, velocity=0.03, force=[], gripper=gripper_value))
            trajectory.add_waypoint(Waypoint(start_up, rotation=start_orientation, velocity=0.03, force=[], gripper=gripper_value))
            trajectory.add_waypoint(Waypoint(start_point, rotation=start_orientation, velocity=0.04, force=[], gripper=gripper_value))
            
        return trajectory
            