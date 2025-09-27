import cv2
import json
import numpy as np
import os
from pathlib import Path

from save_load import load_system

PATTERN_SIZE = (13,9)
SQUARE_SIZE = 0.01

if __name__ == "__main__":
    repo_dir = Path(__file__).parent.parent

    config_path = os.path.join(repo_dir, "configs", "feeding_xarm.json")
    robot, head_camera, wrist_camera, head_rotation, wrist_rotation = load_system(config_path)

    camera_intrinsics = head_camera.get_color_intrinsics()
    distortion = np.zeros((5,1))
    objp = np.zeros((PATTERN_SIZE[0] * PATTERN_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN_SIZE[0], 0:PATTERN_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    fk_list = []
    marker_list = []

    while True:
        head_rgb, head_depth = head_camera.get_new_frames()
        head_bgr = cv2.cvtColor(head_rgb, cv2.COLOR_BGR2RGB)
        head_gray = cv2.cvtColor(head_bgr, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(head_gray, PATTERN_SIZE)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(head_gray, corners, (11, 11), (-1, -1), criteria)
            success, rvec, tvec = cv2.solvePnP(objp, corners, camera_intrinsics, distortion)
            cv2.drawChessboardCorners(head_bgr, PATTERN_SIZE, corners, ret)
            cv2.drawFrameAxes(head_bgr, camera_intrinsics, distortion, rvec, tvec, length=0.1)
        cv2.imshow("Head camera", cv2.rotate(head_bgr, head_rotation))
        key = cv2.waitKey(1)
        if key == 13:
            fk = robot.forward_kinematics()
            if success:
                marker_transform = np.eye(4)
                marker_transform[:3, :3] = cv2.Rodrigues(rvec)[0]
                marker_transform[:3, 3] = tvec.flatten()
                fk_list.append(fk)
                marker_list.append(marker_transform)
                cv2.imwrite(os.path.join(repo_dir, "temp", "calibration", f"{len(fk_list)-1}.png"), head_bgr)
            else:
                print("Failed to solve PnP. Try again.")
        elif key == ord("d"):
            if len(fk_list) > 0:
                fk_list.pop()
                marker_list.pop()
                print("Deleted last entry.")
            else:
                print("No entries to delete.")    
        elif key == 27:
            break

    np.save(os.path.join(repo_dir, "temp", "calibration", "gripper_pose.npy"), fk_list)
    np.save(os.path.join(repo_dir, "temp", "calibration", "marker_pose.npy"), marker_list)

    # Alternatively, you can load the saved data from a previous collection
    # fk_list = np.load(os.path.join(repo_dir, "temp", "calibration", "gripper_pose.npy"))
    # marker_list = np.load(os.path.join(repo_dir, "temp", "calibration", "marker_pose.npy"))

    inv_fk_array = np.linalg.inv(np.array(fk_list)) # This inverse is since we are doing an eye-to-hand calibration (i.e. camera is spatially fixed relative to robot base)
    marker_array = np.array(marker_list)

    R_cam2base, t_cam2base = cv2.calibrateHandEye(inv_fk_array[:,:3,:3], inv_fk_array[:,:3,3], marker_array[:,:3,:3], marker_array[:,:3,3], method=cv2.CALIB_HAND_EYE_TSAI)
    print(f"Calibration result:\nRotation:\n{R_cam2base}\nTranslation:\n{t_cam2base.flatten()}")
    transform = np.eye(4)
    transform[:3, :3] = R_cam2base
    transform[:3, 3] = t_cam2base.flatten()

    with open(config_path, "r+") as f:
        data = json.load(f)
        data["transform"] = transform.tolist()
        f.seek(0)
        json.dump(data, f, indent=4)
        f.truncate()
