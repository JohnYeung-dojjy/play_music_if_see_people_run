from __future__ import annotations
import cv2
from os import PathLike
import logging

class StreamInputError(Exception):
    pass

class VideoStreamManager:
    """A Handler for video streams."""
    def __init__(self, camera_id: int|None=None, video_file: PathLike|None=None, fps=30):
        """Initialize the video stream manager

        Args:
            camera_id (int|None,, optional): camera_id to be read from, cannot co-exist with video_file. Defaults to None.
            video_file (PathLike|None, optional): Path for video to be read from, cannot co-exist with camera_id.. Defaults to None.
            fps (int, optional): _description_. Defaults to 30.

        Raises:
            AttributeError: Raised if both camera_id and video_file has non-None inputs
            StreamInputError: Raised if video stream is not loaded successfully
        """        
        if not bool(video_file)^ (bool(camera_id) or camera_id==0): 
            raise AttributeError("Video stream can either come from camera or video file")
        self.camera_id: int|None = camera_id
        self.video_file: PathLike|None = video_file
        logging.info("Preparing video stream")
        self.video_capture = cv2.VideoCapture(camera_id) if camera_id is not None else cv2.VideoCapture(str(video_file))
        logging.info("Done")
        if not self.video_capture.isOpened():
            camera_text = f"camera ID: {camera_id}" if camera_id is not None else ""
            video_text = f"video file: {video_file}" if video_file is not None else ""
            raise StreamInputError(f"Error opening video stream from {camera_text+video_text}")
        # set the video capture frame-per-second
        self.video_capture.set(cv2.CAP_PROP_FPS, fps)
        self.fps = fps
        self.frame_count: int = 0
    
    def __del__(self):
        self.video_capture.release()
        cv2.destroyAllWindows()
    
    def is_confirmed_exit(self)->bool:
        return cv2.waitKey(5) & 0xFF == 27
    
    def read_frames(self):
        """A generator that reads frames from the video stream

        Yields:
            cs2 Mat object: the frame object captured by self.video_capture
        """
        while self.video_capture.isOpened():
            if self.is_confirmed_exit(): break
            ret, frame = self.video_capture.read()
            if not ret:
                if self.camera_id is not None:
                    print("Ignoring empty camera frame.")
                    continue
                else:
                    break
            self.frame_count += 1
            yield frame

def test():
    tester = VideoStreamManager(camera_id=0)
    for frame in tester.read_frames():
        #  {tester.frame_count}
        cv2.imshow(f"test camera input frame", cv2.flip(frame,1))
if __name__ == "__main__":
    test()