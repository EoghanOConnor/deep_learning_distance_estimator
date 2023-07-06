import numpy as np
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
import pandas as pd
import time
import pickle
import tensorflow as tf
import keras
import os
import torch
import cv2

class DistanceEstimator:
    """
    A class to estimate distance of detected objects in a video using yolo and distance models.
    """

    def __init__(self, yolo_model, distance_model, model_predict_func, input_video, output_video, inference_filename):
        """
        Initialize the DistanceEstimator class with the yolo and distance models, 
        prediction function, input video, output video and the filename to save inference times.

        Args:
            yolo_model (object): Object detection model.
            distance_model (object): Distance estimation model.
            model_predict_func (function): Function to make prediction using the distance model.
            input_video (str): File path for the input video.
            output_video (str): File path to save the output video with annotated distances.
            inference_filename (str): File path to save the inference times.

        Attributes:
            cap (object): Capture object to read frames from the video.
            size (tuple): Tuple representing the width and height of the frames.
            out (object): VideoWriter object to write frames to the output video.
            font (object): Font style for the text annotation in frames.
            fontScale (float): Font scale for the text annotation in frames.
            green, red (tuple): Tuple representing the RGB color code.
            thickness (int): Line thickness for rectangle and text annotation in frames.
            inference_times (list): List to save the inference times for each frame.
        """
        self.yolo_model = yolo_model
        self.distance_model = distance_model
        self.predict_func = model_predict_func

        self.cap = cv2.VideoCapture(input_video)
        frame_width = int(self.cap.get(3))
        frame_height = int(self.cap.get(4))
        self.size = (frame_width, frame_height)
        self.out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc('X','V','I','D'), 29, self.size)

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.5
        self.green = (0, 255, 0)
        self.red = (0, 0, 255)
        self.thickness = 2

        self.inference_times = []
        self.inference_filename = inference_filename

    @staticmethod
    def get_inputs(results):
        """
        Static method to extract bounding box coordinates from the yolo model's results.

        Args:
            results (object): Output from the yolo model's prediction.

        Returns:
            list: List of bounding box coordinates.
        """
        try:
            out = results.xyxy[0].numpy()
            return [obj[0:4] for obj in out]
        except:
            print('Empty')
            return []

    def process_video(self):
        """
        Processes each frame in the input video, applies object detection, estimates distance 
        for each detected object, annotates the frame with distances, and writes to the output video.
        Also measures the inference time for each frame.
        """
        max_predictions = 2
        while self.cap.isOpened():
            ret, image = self.cap.read()
            if not ret:
                break

            start_time = time.time()

            try:
                results = self.yolo_model(image)
                yolo_xywh = self.get_inputs(results)

                if len(yolo_xywh)== 0:
                    raise ValueError('Array is empty.')

                for i, detection in enumerate(yolo_xywh):
                    if i >= max_predictions:
                        break

                    xmin, ymin, xmax, ymax = detection
                    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), self.red, self.thickness)

                    prediction = self.predict_func(self.distance_model, detection)
                    org = (int(xmin), int(ymax) + 20)
                    image = cv2.putText(image, "%.2f m" % (prediction), org, self.font,
                                        self.fontScale, self.green, self.thickness, cv2.LINE_AA)

                self.out.write(image)
                end_time = time.time()
                self.inference_times.append(end_time - start_time)

            except Exception as e:
                pass

            cv2.imshow("Distance estimator", image)
            if cv2.waitKey(25) == ord("q"):
                break

        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

        np.save(self.inference_filename, self.inference_times)


def cnn_predict(model, detection):
    """
    Predicts the output of a given detection input.

    This function reshapes the input to fit the model's input shape requirements,
    performs the prediction, and returns the first result.

    Args:
        model (Sequential, Model): The trained model to use for prediction.
        detection (list or numpy array): A list or numpy array containing the bounding box coordinates.
            The coordinates are expected in the following order: xmin, ymin, xmax, ymax.

    Returns:
        float: The predicted output value from the model.
    """
    data = detection.reshape(1, 4, 1)
    return model.predict(data)[0]

def main():
    yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
    yolo_model.classes = 0
    cnn_model = tf.keras.models.load_model("cnn_model/")
    estimator = DistanceEstimator(yolo_model, cnn_model, cnn_predict, 'limerick_mid_far_range.mp4', 'cnn_initial_mid_far.avi', 'inference_cnn.npy')
    estimator.process_video()

    
if __name__ == "__main__":
    main()
