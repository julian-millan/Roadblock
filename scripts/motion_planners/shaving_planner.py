import numpy as np
from scipy.spatial.transform import Rotation

from robots import BaseRobot
from trajectories import Trajectory, Waypoint
from vision.cameras import BaseCamera
from vision.pose_detector import PoseDetector

class ShavingPlanner:
    def __init__(self):
        pass

    def shaving_orientation_fn(self, motion_dir):
        def thunk(surface_normal):
            y_dir = np.cross(surface_normal, motion_dir)
            y_dir = y_dir / np.linalg.norm(y_dir)
            orientation = np.array([-surface_normal, y_dir, np.cross(-surface_normal, y_dir)]).T
            orientation = orientation @ Rotation.from_euler('y', 20, degrees=True).as_matrix()
            return orientation
        return thunk

    def plan_trajectory(self, robot: BaseRobot, camera: BaseCamera, detector: PoseDetector, side="left", trajectory_num=1) -> list[Waypoint]:
        fk = robot.forward_kinematics()
        camera_transform = robot.get_camera_transform()
        gripper_value = robot.get_gripper_value()
        start_point = fk[0:3,3]
        start_orientation = fk[0:3,0:3]
        wrist_pixel = detector.query_landmark(side + " wrist")
        elbow_pixel = detector.query_landmark(side + " elbow")
        wrist_point = camera.get_3d_from_pixel(pixel=wrist_pixel, transform=camera_transform)
        elbow_point = camera.get_3d_from_pixel(pixel=elbow_pixel, transform=camera_transform)

        shaving_start_pixel = (0.3*wrist_pixel + 0.7*elbow_pixel).astype(int)
        shaving_end_pixel = (0.9*wrist_pixel + 0.1*elbow_pixel).astype(int)
        shaving_start = camera.get_3d_from_pixel(shaving_start_pixel, transform=camera_transform)
        shaving_end = camera.get_3d_from_pixel(shaving_end_pixel, transform=camera_transform)
        
        forearm_dir = (elbow_point - wrist_point) / np.linalg.norm(elbow_point - wrist_point)
        path_pixels = np.linspace(shaving_start_pixel, shaving_end_pixel, 10, dtype=np.uint16, endpoint=True)
        path_normals = [camera.get_normal_from_pixel(pixel=pixel, transform=camera_transform) for pixel in path_pixels]
        shave_start_orientation = self.shaving_orientation_fn(forearm_dir)(np.mean(path_normals, axis=0))

        if trajectory_num == 1:
            shaving_start_up = shaving_start + np.array([0,0,0.08])
            shaving_end_up = shaving_end + np.array([0,0,0.08])

            trajectory = Trajectory()
            trajectory.add_waypoint(Waypoint(start_point, rotation=start_orientation, velocity=0, force=[], gripper=gripper_value))
            trajectory.add_waypoint(Waypoint(shaving_start_up, rotation=shave_start_orientation, velocity=0.03, force=[], gripper=gripper_value))
            trajectory.add_waypoint(Waypoint(shaving_start, rotation=shave_start_orientation, velocity=0.02, force=[], gripper=gripper_value))
            trajectory.add_waypoint(Waypoint(shaving_end, rotation=None, velocity=0.007, force=[1.0, 1.0], gripper=gripper_value, surface_align_fn=self.shaving_orientation_fn(forearm_dir), start_pixel=shaving_start_pixel, end_pixel=shaving_end_pixel))
            trajectory.add_waypoint(Waypoint(shaving_end_up, rotation=None, velocity=0.02, force=[], gripper=gripper_value))
            trajectory.add_waypoint(Waypoint(start_point, rotation=start_orientation, velocity=0.03, force=[], gripper=gripper_value))
        elif trajectory_num == 2:
            contact_pixels = np.linspace(shaving_start_pixel, shaving_end_pixel, 9, endpoint=True)
            contact_pixels = contact_pixels[[0,1,3,5,7,8]]
            single_segment = contact_pixels[1,:] - contact_pixels[0,:]
            for i in range(1,5):
                if i % 2 == 0:
                    contact_pixels[i,:] += np.array([-single_segment[1], single_segment[0]]) * 1.2
                else:
                    contact_pixels[i,:] -= np.array([-single_segment[1], single_segment[0]]) * 1.2
            contact_pixels = contact_pixels.astype(int).tolist()

            shaving_start_up = shaving_start + np.array([0,0,0.05])
            shaving_end_up = shaving_end + np.array([0,0,0.05])

            trajectory = Trajectory()
            trajectory.add_waypoint(Waypoint(start_point, rotation=start_orientation, velocity=0, force=[], gripper=gripper_value))
            trajectory.add_waypoint(Waypoint(shaving_start_up, rotation=shave_start_orientation, velocity=0.03, force=[], gripper=gripper_value))
            trajectory.add_waypoint(Waypoint(shaving_start, rotation=shave_start_orientation, velocity=0.02, force=[], gripper=gripper_value))
            for i in range(len(contact_pixels) - 1):
                trajectory.add_waypoint(Waypoint(
                    camera.get_3d_from_pixel(contact_pixels[i+1], transform=camera_transform),
                    rotation=None, velocity=0.007, force=[1.0, 1.0], gripper=gripper_value,
                    surface_align_fn=self.shaving_orientation_fn(forearm_dir), start_pixel=contact_pixels[i], end_pixel=contact_pixels[i+1]))
            trajectory.add_waypoint(Waypoint(shaving_end_up, rotation=None, velocity=0.02, force=[], gripper=gripper_value))
            trajectory.add_waypoint(Waypoint(start_point, rotation=start_orientation, velocity=0.03, force=[], gripper=gripper_value))

        return trajectory
            