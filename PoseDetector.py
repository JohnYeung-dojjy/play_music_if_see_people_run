from __future__ import annotations # for >3.10 builtin type hints
import cv2
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.pose import Pose, PoseLandmark, POSE_CONNECTIONS
from mediapipe.python.solutions.drawing_styles import get_default_pose_landmarks_style
import numpy as np
import torch
from typing import NamedTuple
import tensorflow as tf
    
class PoseDetectionResult:
    def __init__(self, results: NamedTuple, mask: set=set()):
        # if mask is None:
        #     # discard 0-10 for head positions
        #     mask = {
        #         PoseLandmark.NOSE,
        #         PoseLandmark.LEFT_EYE_INNER,
        #         PoseLandmark.LEFT_EYE,
        #         PoseLandmark.LEFT_EYE_OUTER,
        #         PoseLandmark.RIGHT_EYE_INNER,
        #         PoseLandmark.RIGHT_EYE,
        #         PoseLandmark.RIGHT_EYE_OUTER,
        #         PoseLandmark.LEFT_EAR,
        #         PoseLandmark.RIGHT_EAR,
        #         PoseLandmark.MOUTH_LEFT,
        #         PoseLandmark.MOUTH_RIGHT
        #     }
        # landmark.x, landmark.y are in a [0,1] scale, except when those are out-of-bound
        self.landmarks: list = list([[landmark.x, landmark.y] for (idx, landmark) in enumerate(results.pose_landmarks.landmark) if idx not in mask]) # type: ignore
        
        
    def to_torch_tensor(self):
        return torch.tensor(self.landmarks)
    
    def to_numpy(self):
        return np.array(self.landmarks)

class SinglePersonPoseDetector():
    def __init__(self,
                 static_image_mode=False,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        
        self.pose: Pose = Pose(static_image_mode=static_image_mode,
                         min_detection_confidence=min_detection_confidence,
                         min_tracking_confidence=min_tracking_confidence)
        
    def detect(self, image)->list:
        #TODO: return list of landmarks, instead of image
        image_height, image_width, _ = image.shape
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if results.pose_landmarks: # type: ignore
            for landmark in results.pose_landmarks.landmark: # type: ignore
                landmark_x = min(int(landmark.x * image_width), image_width - 1)
                landmark_y = min(int(landmark.y * image_height), image_height - 1)
                cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), -1)
        draw_landmarks(
        image,
        results.pose_landmarks, # type: ignore
        list(POSE_CONNECTIONS),
        landmark_drawing_spec=get_default_pose_landmarks_style())
        return image
    
def test():
    from VideoStreamManager import VideoStreamManager
    video_stream = VideoStreamManager(camera_id=0)
    pose_detector = SinglePersonPoseDetector()
    for frame in video_stream.read_frames():
        
        cv2.imshow('frame', pose_detector.detect(frame))
    
    


if __name__ == '__main__':
    test()