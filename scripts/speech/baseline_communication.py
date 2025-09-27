from collections import Counter
import copy
import numpy as np

def get_closest_landmark(detected_landmarks, gripper_position):
    landmark_names = ["left wrist", "right wrist",
                      "left elbow", "right elbow",
                      "left shoulder", "right shoulder",
                      "left hip", "right hip",
                      "left knee", "right knee",
                      "left ankle", "right ankle",
                      "left eye", "right eye",
                      "nose", "mouth"]
    landmark_positions = np.zeros((len(landmark_names),3))
    for i, landmark in enumerate(landmark_names):
        if landmark == "mouth":
            landmark_positions[i] = (detected_landmarks["mouth (left)"] + detected_landmarks["mouth (right)"]) / 2
        elif detected_landmarks[landmark] is None:
            landmark_positions[i] = np.array([np.inf, np.inf, np.inf])
        else:
            landmark_positions[i] = detected_landmarks[landmark]

    landmark_ind = np.argmin(np.linalg.norm(landmark_positions - gripper_position, axis=1))
    return landmark_names[landmark_ind]

def get_baseline_communication(segmented_data):
    detected_landmarks = segmented_data["landmark_pos"]
    statements = []
    for trajectory in segmented_data["trajectory"]:
        length = len(trajectory)
        length_50 = int(length * 0.5)
        closest_landmarks = [get_closest_landmark(detected_landmarks, trajectory[i].position) for i in range(length_50, length)]
        closest_landmarks_counter = Counter(closest_landmarks)
        most_common_landmark = closest_landmarks_counter.most_common(1)[0][0]
        statements.append(f"I'm moving towards your {most_common_landmark}.")
    
    return statements