from __future__ import annotations
from multiprocessing import Pool
from glob import glob
from pathlib import Path
import pandas as pd
from PoseDetector import SinglePersonPoseDetector, PoseLandmark
from VideoStreamManager import VideoStreamManager
import logging
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt


def create_training_dataset_from_KTH(cpu_cores: int|None)->None:
  """
  Creates a training dataset from the KTH dataset.
  """
  data = []
  dataset_path = Path("Dataset") / "KTH"
  
  detector = SinglePersonPoseDetector()
  def process_video(video_filename: str, label: str) -> list[list]:
    """
    Processes a single video and returns the detected pose landmarks as a list of tuples.
    """
    nonlocal detector # cannot be passed as argument because the object cannot be store as pickle
    data: list[list] = []
    init_data = [video_filename, label]
    # logging.info(f"processing {video_filename}...")
    video_stream = VideoStreamManager(video_file=video_filename)
    for frame in video_stream.read_frames():
      frame.flags.writeable = False
      image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      results = detector.detect(image)
      # print(results)
      if results is not None: # if there are detected results
        results = results.normalize()
        data.append(init_data + results.to_flattened_list()) # type: ignore
        # print(data[-1])
        # break
    return data
  with Pool(cpu_cores) as pool:
    for label in {"jogging", "walking", "running"}:
      video_filenames = glob(str(dataset_path/label) + "/*.avi")
      results = [pool.apply_async(process_video, (video_filename, label)) for video_filename in video_filenames]
      
      for result in tqdm(results):
        for frame_pose_data in result.get():
          data.append(frame_pose_data)
      
              
  column_names = ["filename", "label"]
  u_x, u_y = '_x', '_y'     
  for enum in PoseLandmark:   
    column_names += [enum.name+u_x, enum.name+u_y]
  dataset = pd.DataFrame(data, columns=column_names)
  dataset.to_csv(f"{dataset_path/'dataset.csv'}", index=False)

if __name__ == "__main__":
  # take multiple arguments from argparse, if argument matches 'KTH', run create_training_dataset_from_KTH()
  import argparse
  parser = argparse.ArgumentParser("Create training dataset csv from a list of datasets")
  parser.add_argument("cores_used", type=int, default=None, help="Number of cores used for multiprocessing. If None, use all available cores")
  parser.add_argument("dataset", nargs='+', help="KTH or other")
  args = parser.parse_args()
  for dataset_name in args.dataset:
    match dataset_name:
      case "KTH":
        create_training_dataset_from_KTH(args.cores_used)
      case _:
        print("Unknown dataset")
        continue
