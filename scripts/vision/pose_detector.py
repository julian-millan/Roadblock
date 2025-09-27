import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.drawing_utils import DrawingSpec
import numpy as np
import os
from pathlib import Path

from vision.cameras import BaseCamera
from vision.utils import rotate

LANDMARK_INDEX = {
    "nose": 0,
    "left eye (inner)": 1,
    "left eye": 2,
    "left eye (outer)": 3,
    "right eye (inner)": 4,
    "right eye": 5,
    "right eye (outer)": 6,
    "left ear": 7,
    "right ear": 8,
    "mouth (left)": 9,
    "mouth (right)": 10,
    "left shoulder": 11,
    "right shoulder": 12,
    "left elbow": 13,
    "right elbow": 14,
    "left wrist": 15,
    "right wrist": 16,
    "left pinky": 17,
    "right pinky": 18,
    "left index": 19,
    "right index": 20,
    "left thumb": 21,
    "right thumb": 22,
    "left hip": 23,
    "right hip": 24,
    "left knee": 25,
    "right knee": 26,
    "left ankle": 27,
    "right ankle": 28,
    "left heel": 29,
    "right heel": 30,
    "left foot index": 31,
    "right foot index": 32,
}

MAJOR_LANDMARKS = [
    "left shoulder", "right shoulder",
    "left elbow", "right elbow",
    "left wrist", "right wrist",
    "left hip", "right hip",
    "left knee", "right knee",
    "left ankle", "right ankle",
]

class PoseDetector(object):
    def __init__(self):
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        repo_dir = Path(__file__).parent.parent.parent
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=os.path.join(repo_dir, "resources", "pose_landmarker_full.task")),
            running_mode=VisionRunningMode.IMAGE)
        self.landmarker = PoseLandmarker.create_from_options(options)
        self.pose_landmarker_result = None
        self.rotation = None

    def detect_landmarks(self, rgb_image, rotation=None):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rotate(rgb_image, rotation))
        self.rotation = rotation
        self.pose_landmarker_result = self.landmarker.detect(mp_image)
        self.image_width = mp_image.width
        self.image_height = mp_image.height

    def query_landmark(self, landmark_name):
        if self.pose_landmarker_result is None:
            print("Please run detect_landmarks first on an image, and then query the landmark.")
            return None
        if landmark_name not in LANDMARK_INDEX.keys():
            raise RuntimeError(f"The specified landmark name '{landmark_name}' does not exist.")
        idx = LANDMARK_INDEX[landmark_name]
        landmarks = self.pose_landmarker_result.pose_landmarks[0]
        if landmarks[idx].presence < 0.8 or landmarks[idx].visibility < 0.8:
            return None
        pixel = np.array([round(landmarks[idx].x * self.image_width), round(landmarks[idx].y * self.image_height)])
        if self.rotation == cv2.ROTATE_90_CLOCKWISE:
            pixel = np.array([pixel[1], self.image_width-1-pixel[0]])
        elif self.rotation == cv2.ROTATE_180:
            pixel = np.array([self.image_width-1-pixel[0], self.image_height-1-pixel[1]])
        elif self.rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
            pixel = np.array([self.image_height-1-pixel[1], pixel[0]])
        return pixel

    def draw_landmarks_on_image(self, rgb_image):
        pose_landmarks_list = self.pose_landmarker_result.pose_landmarks
        if self.rotation is not None:
            annotated_image = np.copy(cv2.rotate(rgb_image, self.rotation))
            reverse_rotation = self.rotation
            if reverse_rotation == cv2.ROTATE_90_CLOCKWISE:
                reverse_rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
            elif reverse_rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
                reverse_rotation = cv2.ROTATE_90_CLOCKWISE
        else:
            annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            drawing_style = solutions.drawing_styles.get_default_pose_landmarks_style()
            for major_landmark in MAJOR_LANDMARKS:
                current_spec = drawing_style[LANDMARK_INDEX[major_landmark]]
                new_spec = DrawingSpec(current_spec.color, thickness=-1, circle_radius=15)
                drawing_style[LANDMARK_INDEX[major_landmark]] = new_spec
            connections = solutions.pose.POSE_CONNECTIONS
            major_landmark_indices = [LANDMARK_INDEX[major_landmark] for major_landmark in MAJOR_LANDMARKS]
            connections_drawing_style = {}
            for connection in connections:
                if connection[0] in major_landmark_indices and connection[1] in major_landmark_indices:
                    connections_drawing_style[connection] = DrawingSpec(thickness=10)
                else:
                    connections_drawing_style[connection] = DrawingSpec()
            solutions.drawing_utils.draw_landmarks(
                image = annotated_image,
                landmark_list = pose_landmarks_proto,
                connections = connections,
                landmark_drawing_spec = drawing_style,
                connection_drawing_spec = connections_drawing_style)
        if self.rotation is not None:
            return cv2.rotate(annotated_image, reverse_rotation)
        else:
            return annotated_image
    
    def face_blurring(self, rgb_image):
        nose_pixel = self.query_landmark("nose")
        lmouth_pixel = self.query_landmark("mouth (left)")
        rmouth_pixel = self.query_landmark("mouth (right)")
        leye_pixel = self.query_landmark("left eye")
        reye_pixel = self.query_landmark("right eye")
        if nose_pixel is None or lmouth_pixel is None or rmouth_pixel is None or leye_pixel is None or reye_pixel is None:
            raise RuntimeError("Could not do face blurring! Please make sure the entire face is visible.")
        [mx, my, _, _] = cv2.fitLine(np.array([nose_pixel, (lmouth_pixel+rmouth_pixel)/2, (leye_pixel+reye_pixel)/2]), cv2.DIST_L2, 0, 0.01, 0.01)
        angle = float(np.arctan2(my, mx) * 180 / np.pi)
        dist = max(np.linalg.norm(np.subtract(leye_pixel, nose_pixel)), np.linalg.norm(np.subtract(reye_pixel, nose_pixel)))
        blurred_image = cv2.GaussianBlur(rgb_image, (77, 77), 100)
        color_face = np.copy(rgb_image)
        cv2.ellipse(img=color_face, center=nose_pixel, axes=(int(dist*2), int(dist*1.5)), angle=angle, startAngle=0, endAngle=360, color=(0,0,255), thickness=-1)
        blur_mask = np.all(color_face == np.array([0,0,255]), axis=-1)[:,:,None]
        return np.where(blur_mask, blurred_image, rgb_image)
    
    def face_normal(self, camera: BaseCamera):
        facial_keypoints = [self.query_landmark("nose"), self.query_landmark("left eye"), self.query_landmark("right eye"), self.query_landmark("mouth (left)"), self.query_landmark("mouth (right)")]
        facial_keypoints_3d = np.array([camera.get_3d_from_pixel(pixel=keypoint) for keypoint in facial_keypoints])
        _, _, vh = np.linalg.svd(facial_keypoints_3d - np.mean(facial_keypoints_3d, axis=0))
        normal = vh[-1]
        if np.linalg.norm(facial_keypoints_3d[0]) < np.linalg.norm(facial_keypoints_3d[0] + normal*0.001):
            normal *= -1
        return normal
    
    def query_all_landmarks_3d(self, camera: BaseCamera, transform: np.ndarray):
        width, height = camera.width, camera.height
        keypoints = [self.query_landmark(landmark_name) for landmark_name in LANDMARK_INDEX.keys()]
        keypoints_3d = [None if keypoint is None or keypoint[0] > width or keypoint[0] < 0 or keypoint[1] > height or keypoint[1] < 0 else camera.get_3d_from_pixel(pixel=keypoint, transform=transform) for keypoint in keypoints]
        output = {name: point for (name, point) in zip(LANDMARK_INDEX.keys(), keypoints_3d)}
        return output
