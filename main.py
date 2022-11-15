from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import tensorflow as tf
from official.vision.ops.preprocess_ops import normalize_image
from PIL import Image
from six import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pathlib
import cv2
import logging
logging.disable(logging.WARNING)

plastic_model = 'models/saved_model/'


def load_image_into_numpy_array(image):
    (im_width, im_height, channel) = image.shape
    return image.astype(np.uint8)


def build_inputs_for_segmentation(image):
    image = normalize_image(image)
    return image


model_display_name = 'plastic_model'
model_handle = model_display_name
print('Selected model:' + model_display_name)
print('Model Handle at TensorFlow Hub: {}'.format(model_handle))
