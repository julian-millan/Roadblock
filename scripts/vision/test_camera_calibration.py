import cv2
import numpy as np
import os
from pathlib import Path

from save_load import load_system

if __name__ == "__main__":
    repo_dir = Path(__file__).parent.parent

    config_path = os.path.join(repo_dir, "configs", "feeding_xarm.json")
    robot, head_camera, wrist_camera, tts_engine, vlm_client = load_system(config_path)

    camera_intrinsics = head_camera.get_color_intrinsics()
    camera_transform = robot.get_camera_transform()
    distortion = np.zeros((5,1))

    while True:
        head_rgb, head_depth = head_camera.get_new_frames()
        fk = robot.forward_kinematics()
        gripper_in_camera = robot.convert_base_to_camera(fk[:3,3], camera_transform)
        gripper_pixel = camera_intrinsics @ gripper_in_camera / gripper_in_camera[2]
        gripper_pixel = gripper_pixel[:2].astype(int)
        cv2.circle(head_rgb, tuple(gripper_pixel), 5, (0, 255, 0), -1)
        cv2.imshow("Head camera", cv2.rotate(cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR), head_camera.rotation))
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()
    head_camera.close()
    wrist_camera.close()
