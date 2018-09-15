# -*- coding: utf-8 -*-

import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import ast
import os
from yolo.darknet53 import Darknet53
from yolo.weights import WeightReader
from yolo import PROJECT_ROOT

WEIGHT_FILE = os.path.join(os.path.dirname(PROJECT_ROOT), "dataset", "yolo", "darknet53.weights")
LABEL_FILE = os.path.join(PROJECT_ROOT, "imagenet_labels.txt")
IMG_FILE = os.path.join(PROJECT_ROOT, "imgs", "14_bird.jpg")


if __name__ == '__main__':
    # 1. create darknet model
    darknet = Darknet53()
 
    # 2. load pretrained weights
    reader = WeightReader(WEIGHT_FILE)
    reader.load_weights(darknet)
    
    # 3. inference
    import cv2
    img = cv2.imread(IMG_FILE)
    img = cv2.resize(img, (416,416))          #Resize to the input dimension
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    imgs = np.expand_dims(img, axis=0).astype(np.float32)
    y = darknet(tf.constant(imgs))[0]
    label = np.argmax(y)
    with open(LABEL_FILE) as imagenet_classes_file:
        decoding_dict = ast.literal_eval(imagenet_classes_file.read())
    print(label, decoding_dict[label])


