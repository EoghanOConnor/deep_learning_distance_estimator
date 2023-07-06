# The CNN model is created
# The MAE and R2 score after training displayed to show performance prior to Monocular vision.
# The models are saved locally.

import numpy as np
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error
import pandas as pd
import time
import pickle
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import  EarlyStopping
import os
import torch
import cv2

class ModelWrapper:
    """
    This class serves as a wrapper for any machine learning model. 
    It allows for the training, performance evaluation and saving of the model.
    """

    def __init__(self, model, filename):
        """
        Initializes the ModelWrapper class with a model and filename for saving the model.

        Args:
            model (object): Machine learning model object
            filename (str): File name where the trained model will be saved

        Attributes:
            model (object): Machine learning model object
            filename (str): File name where the trained model will be saved
            inputs_labels (pd.DataFrame): DataFrame containing the input features and labels
            inputs (pd.DataFrame): DataFrame containing the input features
            labels (pd.Series): Series containing the labels
            X_train (pd.DataFrame): Training input features
            X_test (pd.DataFrame): Testing input features
            y_train (pd.Series): Training labels
            y_test (pd.Series): Testing labels
            early_stopping (EarlyStopping): Early stopping configuration to avoid overfitting
        """
        self.model = model
        self.filename = filename
        self.inputs_labels = pd.read_csv('pedestrians.csv')
        self.inputs = self.inputs_labels[['xmin', 'ymin', 'xmax', 'ymax']]
        self.labels = self.inputs_labels['zloc']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.inputs, self.labels, test_size=0.10, random_state=42)
        self.early_stopping = EarlyStopping(monitor='loss', patience=20)
        
    def train(self):
        """
        Trains the model using the input data. 
        For Sequential models, the training includes an early stopping callback.
        """
        if type(self.model).__name__ == "Sequential":
            self.model.fit(self.X_train, self.y_train, verbose = 1, epochs=130, callbacks=[self.early_stopping])
        else:
            self.model.fit(self.X_train, self.y_train)

    def performance(self):
        """
        Evaluates the performance of the model on the test data.
        Prints the Mean Absolute Error (MAE) and the R2 score.
        """
        prediction = self.model.predict(self.X_test)
        print(f"MAE: {mean_absolute_error(self.y_test, prediction)}")
        print(f"R2 Score: {r2_score(self.y_test, prediction)}")
    
    def save(self):
        """
        Saves the trained model to the file specified in self.filename
        """
        pickle.dump(self.model, open(self.filename,'wb'))

def main():
    # Initialize CNN
    regr_cnn = tf.keras.models.Sequential([
        tf.keras.layers.Reshape((4, 1), input_shape=(4,)),
        tf.keras.layers.Conv1D(64, 2, activation='relu'),
        tf.keras.layers.Conv1D(64, 1, activation='relu'),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    # Compile CNN
    regr_cnn.compile(loss = 'mean_squared_error', optimizer='adam')
    wrapper_cnn = ModelWrapper(regr_cnn, 'cnn.sav')
    wrapper_cnn.train()
    wrapper_cnn.performance()

    # Save weights
    regr_cnn.save_weights("cnn_model_weights")
    regr_cnn.save("cnn_model")

if __name__ == "__main__":
    main()
