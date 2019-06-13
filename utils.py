import numpy as np

from keras import backend as K
from keras.applications.densenet import preprocess_input as preprocess_input_torch
from keras.applications.mobilenet import preprocess_input as preprocess_input_tf
from keras.applications.vgg16 import preprocess_input as preprocess_input_caffe
from keras.preprocessing import image


def deg2rad(deg):
    pi_on_180 = 0.017453292519943295
    return deg * pi_on_180


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def NP2IMG(img, path):
    img = image.array_to_img(img)
    img.save(path)
    # img.show()


def IMG2NP(img_path, img_h, img_w):
    img = image.load_img(img_path, target_size=(img_h, img_w))
    input_img_data = image.img_to_array(img).astype('uint8')  # from PIL format to numpy format
    return input_img_data


def NP2DNNin(img, diff_preprocess='tf'):
    input_img_data = np.expand_dims(img, axis=0).astype(K.floatx())
    if diff_preprocess == 'caffe':
        input_img_data = preprocess_input_caffe(input_img_data)
    elif diff_preprocess == 'tf':
        input_img_data = preprocess_input_tf(input_img_data)
    elif diff_preprocess == 'torch':
        input_img_data = preprocess_input_torch(input_img_data)
    else:
        print('No such preprocess mode: {}'.format(diff_preprocess))
    return input_img_data


def filter_result(categories, bboxes, scores):
    return categories[scores > .5], bboxes[scores > .5]


def violate(categories_true, bboxes_true, categories_predict, bboxes_predict, IMG_H, IMG_W, iou_threshold=.5):
    # checking bbox false negative
    for bbox_true in bboxes_true:
        violate = 0
        for bbox_predict in bboxes_predict:
            iou = bb_intersection_over_union(bbox_true, bbox_predict, IMG_H, IMG_W)
            if iou < iou_threshold:
                violate += 1
        # does not match any ground truth bounding box
        if violate == len(bboxes_predict):
            print('violate bbox false negative')
            return True

    # checking bbox false positive
    for bbox_predict in bboxes_predict:
        violate = 0
        for bbox_true in bboxes_true:
            iou = bb_intersection_over_union(bbox_true, bbox_predict, IMG_H, IMG_W)
            if iou < iou_threshold:
                violate += 1
        # does not match any ground truth bounding box
        if violate == len(bboxes_true):
            print('violate bbox false positive')
            return True

    # checking categories false negative
    for category_true in categories_true:
        if category_true not in categories_predict:
            print('violate category false negative')
            return True

    # checking categories false positive
    for category_predict in categories_predict:
        if category_predict not in categories_true:
            print('violate category false positive')
            return True


def bb_intersection_over_union(box_t, box_p, IMG_H, IMG_W):
    # Ground truth is xmin, ymin, xmax, ymax
    # Predicted is ymin, xmin, ymax, xmax, and are normalized
    xmint, ymint, xmaxt, ymaxt = box_t[0], box_t[1], box_t[2], box_t[3]
    yminp, xminp, ymaxp, xmaxp = box_p[0] * IMG_H, box_p[1] * IMG_W, box_p[2] * IMG_H, box_p[3] * IMG_W

    xI1 = max(xmint, xminp)
    yI1 = max(ymint, yminp)

    xI2 = min(xmaxt, xmaxp)
    yI2 = min(ymaxt, ymaxp)

    inter_area = max(0, xI2 - xI1 + 1) * max(0, yI2 - yI1 + 1)

    bboxt_area = (xmaxt - xmint + 1) * (ymaxt - ymint + 1)
    bboxp_area = (xmaxp - xminp + 1) * (ymaxp - yminp + 1)

    union = (bboxt_area + bboxp_area) - inter_area

    return inter_area / union
