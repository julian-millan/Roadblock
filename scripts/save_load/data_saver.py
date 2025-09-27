import copy
import os
import pickle
import numpy as np
from scipy.spatial.transform import Rotation

from trajectories import Trajectory

def save_data(directory: str, filename: str, trajectory: Trajectory, landmark_pos: dict, camera_transform: np.ndarray, camera_intrinsics: np.ndarray, 
              raw_rgb: np.ndarray, annotated_rgb: np.ndarray, depth: np.ndarray, rgb_wrist: np.ndarray, head_rotation: int, wrist_rotation:int, save=True):
    output = dict()
    output["trajectory"] = copy.deepcopy(trajectory)
    # Can't save surface align function since it's local
    for w in output["trajectory"]:
        w.surface_align_fn = None
    output["camera_intrinsics"] = camera_intrinsics
    output["T_base_camera_color"] = [camera_transform for i in range(len(trajectory))]
    output["initial_rgb"] = annotated_rgb
    output["initial_depth"] = depth
    output["raw_rgb"] = raw_rgb
    output["wrist_rgb"] = rgb_wrist
    output["landmark_pos"] = landmark_pos
    output["head_rotation"] = head_rotation
    output["wrist_rotation"] = wrist_rotation

    pose_camera_gripper = []
    for (camera_transform, translation_fk, rotation_fk) in zip(output["T_base_camera_color"], [w.position for w in trajectory], [w.rotation for w in trajectory]):
        camera_gripper_transform = np.linalg.inv(camera_transform) @ np.r_[np.c_[rotation_fk, translation_fk], np.array([[0,0,0,1]])]
        translation = camera_gripper_transform[0:3,3]
        rotation = Rotation.from_matrix(camera_gripper_transform[0:3,0:3])
        pose_camera_gripper.append((translation, rotation.as_quat()))
    output["pose_camera_gripper"] = pose_camera_gripper

    output["gripper_pos_pixel"] = [(camera_intrinsics @ pose[0]/pose[0][2])[0:2] for pose in pose_camera_gripper]

    if save:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, filename), 'wb') as f:
            pickle.dump(output, f)

    return output
