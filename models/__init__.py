from __future__ import annotations
from pathlib import Path
import joblib

def load_pose_action_classifier(*, jogging: str=""): # this function only accepts keyword arguments by adding * in front of all arguments
  """load the pose_action_classifier.
  This function only accepts keyword arguments, usage: load_pose_action_classifier(jogging="running").
  
  If set to 'running', load the classifier which is trained with 'jogging' treated as 'running' in the dataset.
  If set to 'walking', load the classifier which is trained with 'jogging' treated as 'walking' in the dataset.
  If set to ''       , load the classifier which is trained with 'jogging' treated as ''        in the dataset.
  
  Args:
    jogging: "running", "walking", or "" (no jogging)
  
  Returns:
    the loaded classifier (sklearn.svm.SVC)
  """
  current_dir = Path(__file__).parent.absolute()
  if jogging == "":
    target = "pose_action_classifier_no_jogging.pkl"
  elif jogging == "running":
    target = "pose_action_classifier_jogging=running.pkl"
  elif jogging == "walking":
    target = "pose_action_classifier_jogging=walking.pkl"
  else:
    raise ValueError("Invalid value for jogging.")
  return joblib.load(current_dir/"action"/target)

def load_label_encoder():
  """Loads a label encoder (running or walking)"""
  
  current_dir = Path(__file__).parent.absolute()
  return joblib.load(current_dir/"label"/"label_encoder.pkl")