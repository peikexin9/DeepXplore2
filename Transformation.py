import itertools

import numpy as np
import tensorflow as tf
from tensorflow.contrib.image import rotate, translate

from utils import deg2rad, angle_between
import cv2


class Transformation:
    def __init__(self, transform_type, img_h=1200, img_w=1920):
        self.transform_type = transform_type
        self.img_h = img_h
        self.img_w = img_w

        if self.transform_type == 'shift':
            self.states = self.shift_states
            self.transform = self.shift
        elif self.transform_type == 'rot':
            self.states = self.rot_states
            self.transform = self.rot
        elif self.transform_type == 'bright':
            self.states = self.bright_states
            self.transform = self.bright
        elif self.transform_type == 'ctra':
            self.states = self.contrast_states
            self.transform = self.contrast
        elif self.transform_type == 'avg_blur':
            self.states = self.avg_blur_states
            self.transform = self.avg_blur
        elif self.transform_type == 'med_blur':
            self.states = self.med_blur_states
            self.transform = self.med_blur

    def rot(self, state, img):
        return rotate(img.astype(np.float32), state, name='rot_image_tensor')

    def rot_states(self, bound=1.1):
        all_degrees = set()

        # calculate how many possible coordinates with different radius
        coordinates = []
        # all possible ending point of a radius
        for i in np.arange(.5, self.img_h / 2, 1.):
            for j in np.arange(.5, self.img_w / 2, 1.):
                coordinates.append(np.array([i, j]))

        # add critical rotation degree
        for orig_coordinate in coordinates:
            for horizontal_threshold_line in np.arange(-self.img_h / 2. + 1, self.img_h / 2., 1.):
                radius_squared = np.sum(orig_coordinate ** 2)
                new_coordinate_y_squared = horizontal_threshold_line ** 2
                if radius_squared > new_coordinate_y_squared:  # has intersection with the line
                    new_coordinate = np.array(
                        [np.sqrt(radius_squared - new_coordinate_y_squared), horizontal_threshold_line])
                    degree = angle_between(orig_coordinate, new_coordinate) / np.pi * 180

                    if bound >= degree:
                        all_degrees.add(degree)
                        all_degrees.add(-degree)

            for vertical_threshold_line in np.arange(-self.img_w / 2. + 1, self.img_w / 2., 1.):
                radius_squared = np.sum(orig_coordinate ** 2)
                new_coordinate_x_squared = vertical_threshold_line ** 2
                if radius_squared > new_coordinate_x_squared:  # has intersection with the line
                    new_coordinate = np.array(
                        [vertical_threshold_line, np.sqrt(radius_squared - new_coordinate_x_squared)])
                    degree = angle_between(orig_coordinate, new_coordinate) / np.pi * 180

                    if bound >= degree:
                        all_degrees.add(degree)
                        all_degrees.add(-degree)

        return list(sorted(list(all_degrees)))

    def bright(self, state, img):
        return tf.clip_by_value(img.astype(np.float32) + state, 0, 255)

    def bright_states(self, bound=100):
        return np.arange(-bound, bound).astype(np.float32)

    def shift(self, state, img):
        return translate(img.astype(np.float32), state)

    def shift_states(self, bound_h=10, bound_w=10):
        xs = np.arange(-bound_h, bound_h)
        ys = np.arange(-bound_w, bound_w)

        return np.array(list(itertools.product(xs, ys)), dtype=np.float32)

    def contrast(self, state, img):
        return tf.clip_by_value(img.astype(np.float32) * state, 0, 255)

    def contrast_states(self, bound=1.5):
        all_ctra_parms = []
        for i in np.arange(1, 255, 1):
            for j in np.arange(.5, 255.5, 1):
                if 1. / bound <= j / i <= bound:
                    all_ctra_parms.append(j / i)

        return np.array(all_ctra_parms, dtype=np.float32)

    def avg_blur_cv2(self, state, img):
        img_np = img.numpy()
        state_np = state.numpy()
        return cv2.blur(img_np, (state_np, state_np))

    def avg_blur(self, state, img):
        return tf.py_function(self.avg_blur_cv2, [state, img], tf.float32)

    def avg_blur_states(self, bound=6):
        avg_blur_parms = []
        for i in np.arange(1, bound):
            avg_blur_parms.append(i)

        return np.array(avg_blur_parms)

    def med_blur_cv2(self, state, img):
        img_np = img.numpy()
        state_np = state.numpy()
        return cv2.medianBlur(img_np, state_np)

    def med_blur(self, state, img):
        return tf.py_function(self.med_blur_cv2, [state, img], tf.float32)

    def med_blur_states(self, bound=13):
        med_blur_parms = []
        for i in np.arange(1, bound, 2):
            med_blur_parms.append(i)

        return np.array(med_blur_parms)
