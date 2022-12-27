import os

import numpy as np
import tensorflow as tf
from tflite_model_maker import image_classifier, model_spec, object_detector
from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader

assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging

logging.set_verbosity(logging.ERROR)


folder_path = "C:\\Users\\sunny\\downloads\\OneDrive_2022-12-01\\Models With Labels_ALL IN ONE FOLDER"


data = DataLoader.from_folder(folder_path)
print(data)
train_data, test_data = data.split(0.75)
model = image_classifier.create(train_data,epochs=5)
loss, accuracy = model.evaluate(test_data)
print(loss);
print(accuracy);
model.export(export_dir='.')