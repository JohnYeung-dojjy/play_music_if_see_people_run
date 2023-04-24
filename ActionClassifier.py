from __future__ import annotations
from PoseDetector import PoseLandmark, PoseDetectionResult
import joblib
from pathlib import Path
from sklearn import svm, preprocessing

MEDIAPIPE_MASK: list[bool] = [False, False, # nose
                              False, False, # left_eye_inner
                              False, False, # left_eye
                              False, False, # left_eye_outer
                              False, False, # right_eye_inner
                              False, False, # right_eye
                              False, False, # right_eye_outer
                              False, False, # left_ear
                              False, False, # right_ear
                              False, False, # mouth_left
                              False, False, # mouth_right
                              True , True , # left_shoulder
                              True , True , # right_shoulder
                              True , True , # left_elbow
                              True , True , # right_elbow
                              True , True , # left_wrist
                              True , True , # right_wrist
                              False, False, # left_pinky
                              False, False, # right_pinky
                              False, False, # left_index
                              False, False, # right_index
                              False, False, # left_thumb
                              False, False, # right_thumb
                              True , True , # left_hip
                              True , True , # right_hip
                              True , True , # left_knee
                              True , True , # right_knee
                              True , True , # left_ankle
                              True , True , # right_ankle
                              False, False, # left_heel
                              False, False, # right_heel
                              False, False, # left_foot_index
                              False, False, # right_foot_index
                              ]

class PoseActionClassifier:
  def __init__(self):
    self.model: svm.SVC = joblib.load(str(Path('models')/'action'/'pose_action_classifier.pkl'))
    self.label_encoder: preprocessing.LabelEncoder = joblib.load(str(Path('models')/'label'/'label_encoder.pkl'))

  def classify(self, pose: PoseDetectionResult)->str:
    data = pose.normalize().np_landmarks.flatten()[MEDIAPIPE_MASK].reshape(1, -1) # reshape as it contains only 1 sample
    # print(data.shape)
    # data = data
    # print(data.shape)
    # data = data[MEDIAPIPE_MASK]
    # print(data.shape)
    y_pred = self.model.predict(data)
    label = self.label_encoder.inverse_transform(y_pred)
    print(y_pred, label)
    return label[0]
    