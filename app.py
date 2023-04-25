"""Main program for the play audio when see person running"""
from __future__ import annotations
from .PoseDetector import SinglePersonPoseDetector, PoseDetectionResult
from .VideoStreamManager import VideoStreamManager
from .AudioPlayer import AudioPlayer
from .ActionClassifier import PoseActionClassifier
from os import PathLike
import time

def main(camera_id: int|str|None=None, 
         video_file: PathLike|str|None=None,
         fps: int=30,
         display_image: bool=False,
         display_landmarks: bool=False):
  """play audio if detected people running in camera/video
  1 detect pose in frame (from camera/video)
  2 if pose is not detected (detected_pose is None) then
      - pause audio stream
    else 
      - detect action from pose information (walking, jogging, running)
      - if action is running then
          - resume (play) audio stream
        else
          - pause audio stream

  Args:
      camera_id (int | str | None, optional): _description_. Defaults to None.
      video_file (PathLike | str | None, optional): _description_. Defaults to None.
      fps (int, optional): _description_. Defaults to 30.
      display_image (bool, optional): _description_. Defaults to False.
      display_landmarks (bool, optional): _description_. Defaults to False.
  """
  audio_player = AudioPlayer()
  video_stream = VideoStreamManager(camera_id=camera_id,
                                    video_file=video_file,
                                    fps=fps)
  pose_detector = SinglePersonPoseDetector()
  action_classifier = PoseActionClassifier(jogging="walking")
  
  last_resume_time = time.time()
  
  for frame in video_stream.read_frames():
    img=frame.copy()
    detected_pose: PoseDetectionResult|None = pose_detector.detect(img)
                                                                  #  display_image=display_image,
                                                                  #  display_landmarks=display_landmarks)
    if detected_pose is None: # continue to next frame if no pose detected
      audio_player.pause()
      action = ""
      # continue  
    else:
      action = action_classifier.classify(detected_pose)
      if action == "running":
        audio_player.resume()
        last_resume_time = time.time()
      else:
        current_time = time.time()
        if current_time - last_resume_time < 1: continue
        #  last_resume_time = current_time
        audio_player.pause()
        # audio_player.resume()

    action_classifier.display_image(img, action, detected_pose, display_image, display_landmarks)

if __name__ == "__main__":
  main(camera_id=0,
       # video_file="Dataset/KTH/running/person02_running_d1_uncomp.avi",
       # fps=60,
       display_image=True,
       display_landmarks=True)