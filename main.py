import logging
import cv2
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
import tensorflow as tf

from PIL import Image
from six import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
logging.disable(logging.WARNING)

plastic_model = 'models/saved_model/'
PATH_TO_LABELS = 'plastic_type_labels.pbtxt'


def normalize_image(image,
                    offset=(0.485, 0.456, 0.406),
                    scale=(0.229, 0.224, 0.225)):
    with tf.name_scope('normalize_image'):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        offset = tf.constant(offset)
        offset = tf.expand_dims(offset, axis=0)
        offset = tf.expand_dims(offset, axis=0)
        image -= offset

        scale = tf.constant(scale)
        scale = tf.expand_dims(scale, axis=0)
        scale = tf.expand_dims(scale, axis=0)
        image /= scale
        return image


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (1, im_height, im_width, 3)).astype(np.uint8)


def build_inputs_for_segmentation(image):
    image = normalize_image(image)
    return image


model_display_name = 'plastic_types_model/saved_model/saved_model/'
model_handle = model_display_name
print('Selected model:' + model_display_name)
print('Model Handle at TensorFlow Hub: {}'.format(model_handle))
print('Labels selected for', model_display_name)
print('\n')
category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)
print(category_index)
print('loading model...')
model = tf.saved_model.load(model_handle)
print('model loaded!')


flip_image_horizontally = False
convert_image_to_grayscale = False

image = "image_2.png"
image_path = Image.open(image)
image_path = image_path.convert('RGB')
image_np = load_image_into_numpy_array(image_path)

if (flip_image_horizontally):
    image_np[0] = np.fliplr(image_np[0]).copy()

if (convert_image_to_grayscale):
    image_np[0] = np.tile(
        np.mean(image_np[0], 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

print('min:', np.min(image_np[0]), 'max:', np.max(image_np[0]))
detection_fn = model.signatures['serving_default']
height = detection_fn.structured_input_signature[1]['inputs'].shape[1]
width = detection_fn.structured_input_signature[1]['inputs'].shape[2]
input_size = (height, width)
print(input_size)
image_np_cp = cv2.resize(
    image_np[0], input_size[::-1], interpolation=cv2.INTER_AREA)
image_np = build_inputs_for_segmentation(image_np_cp)
image_np = tf.expand_dims(image_np, axis=0)
image_np.get_shape()
results = detection_fn(image_np)
result = {key: value.numpy() for key, value in results.items()}
print(result.keys())
label_id_offset = 0
min_score_thresh = 0.6
use_normalized_coordinates = True

if use_normalized_coordinates:
    # Normalizing detection boxes
    result['detection_boxes'][0][:, [0, 2]] /= height
    result['detection_boxes'][0][:, [1, 3]] /= width

if 'detection_masks' in result:
    # we need to convert np.arrays to tensors
    detection_masks = tf.convert_to_tensor(result['detection_masks'][0])
    detection_boxes = tf.convert_to_tensor(result['detection_boxes'][0])

    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
        detection_masks, detection_boxes,
        image_np.shape[1], image_np.shape[2])
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       np.uint8)

    result['detection_masks_reframed'] = detection_masks_reframed.numpy()
viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_cp,
    result['detection_boxes'][0],
    (result['detection_classes'][0] + label_id_offset).astype(int),
    result['detection_scores'][0],
    category_index=category_index,
    use_normalized_coordinates=use_normalized_coordinates,
    max_boxes_to_draw=200,
    min_score_thresh=min_score_thresh,
    agnostic_mode=False,
    instance_masks=result.get('detection_masks_reframed', None),
    line_thickness=2)

plt.figure(1, figsize=(5, 5))
plt.imshow(image_np_cp)
mask_count = np.sum(result['detection_scores'][0] >= min_score_thresh)
print('Total number of objects found are:', mask_count)
mask = np.zeros_like(detection_masks_reframed[0])
for i in range(mask_count):
    if result['detection_scores'][0][i] >= min_score_thresh:
        mask += detection_masks_reframed[i]

mask = tf.clip_by_value(mask, 0, 1)
plt.figure(2, figsize=(5, 5))
plt.imshow(mask, cmap='gray')
plt.show()
