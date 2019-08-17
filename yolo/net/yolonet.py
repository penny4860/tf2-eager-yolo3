# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import cv2

layers = tf.keras.layers
models = tf.keras.models

from yolo.net.bodynet import Bodynet
from yolo.net.headnet import Headnet
from yolo.net.weights import WeightReader


# Yolo v3
class Yolonet(tf.keras.Model):
    def __init__(self, n_classes=80):
        
        super(Yolonet, self).__init__(name='')
        
        self.body = Bodynet()
        self.head = Headnet(n_classes)

        self.num_layers = 110
        self._init_vars()

    def load_darknet_params(self, weights_file, skip_detect_layer=False):
        weight_reader = WeightReader(weights_file)
        weight_reader.load_weights(self, skip_detect_layer)
    
    def predict(self, input_array):
        f5, f4, f3 = self.call(tf.constant(input_array.astype(np.float32)))
        return f5.numpy(), f4.numpy(), f3.numpy()

    def call(self, input_tensor, training=False):
        s3, s4, s5 = self.body(input_tensor, training)
        f5, f4, f3 = self.head(s3, s4, s5, training)
        return f5, f4, f3

    def get_variables(self, layer_idx, suffix=None):
        if suffix:
            find_name = "layer_{}/{}".format(layer_idx, suffix)
        else:
            find_name = "layer_{}/".format(layer_idx)
        variables = []
        for v in self.variables:
            if find_name in v.name:
                variables.append(v)
        return variables

    def _init_vars(self):
        sample = tf.constant(np.random.randn(1, 224, 224, 3).astype(np.float32))
        self.call(sample, training=False)


def preprocess_input(image, net_size):
    """
    # Args
        image : array, shape of (H, W, 3)
            RGB-ordered
        net_size : int
    """
    # resize the image to the new size
    preprocess_img = cv2.resize(image/255., (net_size, net_size))
    return np.expand_dims(preprocess_img, axis=0)


if __name__ == '__main__':
    inputs = tf.constant(np.random.randn(1, 256, 256, 3).astype(np.float32))
    
    # (1, 256, 256, 3) => (1, 8, 8, 1024)
    yolonet = Yolonet()
    f5, f4, f3 = yolonet(inputs)
    print(f5.shape, f4.shape, f3.shape)

    for v in yolonet.variables:
        print(v.name)
