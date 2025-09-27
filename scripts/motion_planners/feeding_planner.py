import numpy as np

from robots import BaseRobot
from trajectories import Trajectory, Waypoint
from vision.cameras import BaseCamera
from vision.pose_detector import PoseDetector

class FeedingPlanner:
    def __init__(self):
        pass

    def plan_trajectory(self, robot: BaseRobot, camera: BaseCamera, detector: PoseDetector, trajectory_num=1) -> Trajectory:
        fk = robot.forward_kinematics()
        camera_transform = robot.get_camera_transform()
        gripper_value = robot.get_gripper_value()
        start_point = fk[0:3,3]
        start_orientation = fk[0:3,0:3]
        mouth_pixel = ((detector.query_landmark("mouth (left)") + detector.query_landmark("mouth (right)")) / 2).astype(int)
        mouth_point = camera.get_3d_from_pixel(mouth_pixel, transform=camera_transform)
        face_normal = detector.face_normal(camera=camera) # gets the normal of the principal face direction, towards the camera
        face_normal_xy = robot.convert_camera_to_base(face_normal, transform_camera_base=camera_transform, rotation_only=True) # this converts the camera frame free vector to base frame
        face_normal_xy[2] = 0 # setting the z-direction to 0 (which should be mostly 0 anyway)
        face_normal_xy /= np.linalg.norm(face_normal_xy) # renormalize
        mouth_out = mouth_point + face_normal_xy * 0.05
        mouth_in = mouth_point - face_normal_xy * 0.01
        mouth_point = mouth_point - np.array([0,0,0.03]) # adjustment if needed

        if trajectory_num == 1:
            trajectory = Trajectory()
            trajectory.add_waypoint(Waypoint(start_point, rotation=start_orientation, velocity=0, force=[], gripper=gripper_value))
            trajectory.add_waypoint(Waypoint(mouth_out, rotation=start_orientation, velocity=0.03, force=[], gripper=gripper_value))
            trajectory.add_waypoint(Waypoint(mouth_in, rotation=start_orientation, velocity=0.008, force=[], gripper=gripper_value))
            trajectory.add_waypoint(Waypoint(mouth_in, rotation=start_orientation, velocity=0.008, force=[], gripper=gripper_value, pause=5))
            trajectory.add_waypoint(Waypoint(mouth_out, rotation=start_orientation, velocity=0.008, force=[], gripper=gripper_value))
            trajectory.add_waypoint(Waypoint(start_point, rotation=start_orientation, velocity=0.04, force=[], gripper=gripper_value))
        elif trajectory_num == 2:
            trajectory = Trajectory()
            trajectory.add_waypoint(Waypoint(start_point, rotation=start_orientation, velocity=0, force=[], gripper=gripper_value))
            trajectory.add_waypoint(Waypoint(mouth_out, rotation=start_orientation, velocity=0.03, force=[], gripper=gripper_value))
            trajectory.add_waypoint(Waypoint(mouth_out, rotation=start_orientation, velocity=0.03, force=[], gripper=gripper_value, pause=10))
            trajectory.add_waypoint(Waypoint(start_point, rotation=start_orientation, velocity=0.04, force=[], gripper=gripper_value))

        return trajectory
            