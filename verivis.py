import json
import os
import time

import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils.visualization_utils import visualize_boxes_and_labels_on_image_array

from Transformation import Transformation
from utils import IMG2NP, NP2IMG, violate, filter_result

# Transformation type
transformation_type = 'rot'

# Image index
img_idx = 30

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'dd_model_data/model/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'dd_model_data/model/label_map.pbtxt'

# configurations
threshold_detection = .5
NUM_CLASSES = 4
IMG_H, IMG_W = 1200, 1920
num_parallel_calls = 4
batch_size = 20

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# read a single input
img = IMG2NP(f'dd_model_data/data/image/didi_{img_idx}.jpg', IMG_H, IMG_W)

# Initialize transformation
t = Transformation(transformation_type)
states = t.states()

# Load model for inference
inference_graph = tf.Graph()
with inference_graph.as_default():
    # Create verification set
    states_placeholder = tf.placeholder(states.dtype, states.shape)
    veriset = tf.data.Dataset.from_tensor_slices(states_placeholder)
    veriset = veriset.map(lambda x: t.transform(x, img), num_parallel_calls=num_parallel_calls)
    veriset = veriset.batch(batch_size)
    veriset = veriset.prefetch(buffer_size=4)
    iterator = veriset.make_initializable_iterator()
    next_element = iterator.get_next()

    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        inference_graph_def = tf.GraphDef()
        serialized_graph = fid.read()
        inference_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(inference_graph_def, name='',
                            input_map={
                                'Preprocessor/map/TensorArrayStack/TensorArrayGatherV3:0': next_element})

# Create results folder
if not os.path.exists(f'results/{transformation_type}'):
    os.mkdir(f'results/{transformation_type}')

# Get ground truth
with open('dd_model_data/data/annotation/voyager_image_train.json', 'r') as f:
    meta = json.load(f)
annotations = meta['annotations']
bboxes_true = []
categories_true = []
for annotation in annotations:
    if annotation['image_id'] == img_idx:
        bboxes_true.append(annotation['bbox'])
        categories_true.append(annotation['category_id'])

# Do inference
inference_sess = tf.Session(graph=inference_graph)
inference_sess.run(iterator.initializer, feed_dict={states_placeholder: states})
tensors = [inference_graph.get_tensor_by_name('detection_boxes:0'),
           inference_graph.get_tensor_by_name('detection_scores:0'),
           inference_graph.get_tensor_by_name('detection_classes:0'),
           inference_graph.get_tensor_by_name('num_detections:0'),
           next_element]

# Time it
t = time.time()
i = 0
while True:
    try:
        nms_bboxes, nms_scores, nms_classes, num_detections, images = inference_sess.run(tensors)
        assert nms_bboxes.shape[0] == nms_scores.shape[0] == nms_classes.shape[0] == images.shape[0]
        # Handle if the batch left is smaller than the specified batch size
        if nms_bboxes.shape[0] < batch_size:
            batch_size = nms_bboxes.shape[0]

        for j in range(batch_size):
            # Filter only top predictions for violation checking
            categories_predict, bboxes_predict = filter_result(nms_classes[j], nms_bboxes[j], nms_scores[j])

            # If there is violation
            if violate(categories_true, bboxes_true, categories_predict, bboxes_predict, IMG_H, IMG_W):
                img_viz = visualize_boxes_and_labels_on_image_array(images[j].copy(),
                                                                    nms_bboxes[j],
                                                                    nms_classes[j].astype(np.int32),
                                                                    nms_scores[j],
                                                                    category_index,
                                                                    use_normalized_coordinates=True,
                                                                    max_boxes_to_draw=num_detections[j].astype(
                                                                        np.int32),
                                                                    min_score_thresh=threshold_detection,
                                                                    line_thickness=1)
                NP2IMG(img_viz, f'results/{transformation_type}/{i * batch_size + j}.jpg')
        i += 1

    except tf.errors.OutOfRangeError:
        break

print(f'num_parallel_calls:{num_parallel_calls}, time: {time.time() - t}')
