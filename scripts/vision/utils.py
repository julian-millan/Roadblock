import cv2
import numpy as np

LANDMARK_NAMES = ["left wrist", "right wrist",
                  "left elbow", "right elbow",
                  "left shoulder", "right shoulder",
                  "mouth"]

def get_landmark_pos_from_detections(detected_landmarks):
    landmark_pos = np.zeros((len(LANDMARK_NAMES), 3))
    
    for i, joint_name in enumerate(LANDMARK_NAMES):
        joint_pos = None
        if joint_name != "mouth":
            if detected_landmarks[joint_name] is not None:
                joint_pos = detected_landmarks[joint_name]
        else:
            if detected_landmarks["mouth (left)"] is not None and detected_landmarks["mouth (right)"] is not None:
                mouth_left = detected_landmarks["mouth (left)"]
                mouth_right = detected_landmarks["mouth (right)"]
                joint_pos = (mouth_left + mouth_right) / 2
        if joint_pos is not None:
            landmark_pos[i,:] = joint_pos
        else:
            landmark_pos[i,:] = np.inf*np.ones(3)

    return landmark_pos

def get_distance_to_landmarks(landmark_pos, query_point):
    distances = np.linalg.norm(landmark_pos - query_point, axis=1)
    indices = [i for i, v in sorted(enumerate(distances), key=lambda x: x[1])]
    return [(distances[i], LANDMARK_NAMES[i]) for i in indices]

def rotate(image, rotation):
    if rotation is None:
        return image
    else:
        return cv2.rotate(image, rotation)