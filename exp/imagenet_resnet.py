# -*- coding: utf-8 -*-

import numpy as np
import ast
import os
from yolo import PROJECT_ROOT
import cv2
from keras.applications.resnet50 import ResNet50, preprocess_input

WEIGHT_FILE = os.path.join(os.path.dirname(PROJECT_ROOT), "dataset", "yolo", "darknet53.weights")
LABEL_FILE = os.path.join(PROJECT_ROOT, "imagenet_labels.txt")
IMG_FILE = os.path.join(PROJECT_ROOT, "imgs", "14_bird.jpg")

if __name__ == '__main__':
    resnet = ResNet50(weights='imagenet')
    img = cv2.imread(IMG_FILE)
    img = cv2.resize(img, (224,224))          #Resize to the input dimension
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = preprocess_input(img)
    imgs = np.expand_dims(img, axis=0).astype(np.float32)
    y = resnet.predict(imgs)
    label = np.argmax(y)
       
    with open(LABEL_FILE) as imagenet_classes_file:
        decoding_dict = ast.literal_eval(imagenet_classes_file.read())
    print(label, decoding_dict[label])
    # 14 indigo bunting, indigo finch, indigo bird, Passerina cyanea


