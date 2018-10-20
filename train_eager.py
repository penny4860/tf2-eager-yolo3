# -*- coding: utf-8 -*-

import tensorflow as tf
tf.enable_eager_execution()

import matplotlib.pyplot as plt
import cv2
import os

from yolo.net import Yolonet
from yolo.train import train
from yolo import COCO_ANCHORS, PROJECT_ROOT
from yolo.frontend import YoloDetector
from yolo.utils.box import draw_boxes
from yolo.dataset.generator import create_generator

# {
#     "model" : {
#         "anchors":              [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326],
#         "labels":               ["raccoon"]
#     },
#     "pretrained" : {
#         "keras_format":             "",
#         "darknet_format":           "yolov3.weights",
#     },
#     "train" : {
#         "min_size":             416,
#         "max_size":             416,
#         "num_epoch":            100,
#         "verbose":              1,
#         "train_image_folder":   "tests/dataset/raccoon/imgs/",
#         "train_annot_folder":   "tests/dataset/raccoon/anns/",
#         "batch_size":           2,
#         "learning_rate":        1e-4,
#         "save_folder":         "raccoon",
#         "jitter":               false
#     }
# }



if __name__ == '__main__':
    
    import json
    with open('configs/raccoon.json') as data_file:    
        config = json.load(data_file)
    
    # 1. create generator
    generator = create_generator(config["train"]["train_image_folder"],
                                 config["train"]["train_annot_folder"],
                                 batch_size=config["train"]["batch_size"],
                                 labels_naming=config["model"]["labels"],
                                 anchors=config["model"]["anchors"],
                                 jitter=config["train"]["jitter"])
    # 2. create model
    model = Yolonet(n_classes=len(config["model"]["labels"]))
    model.load_darknet_params(config["pretrained"]["darknet_format"], skip_detect_layer=True)
 
    # 3. define optimizer    
    optimizer = tf.train.AdamOptimizer(learning_rate=float(["train"]["learning_rate"]))
        
    # 4. training
    train(generator,
          optimizer,
          model,
          num_epoches=config["train"]["num_epoch"],
          verbose=1,
          save_dname=config["train"]["save_folder"])

    # 5. prepare sample images
    import glob
    img_fnames = glob.glob(os.path.join(config["train"]["train_image_folder"], "*.*"))
    imgs = [cv2.imread(fname)[:,:,::-1] for fname in img_fnames]

    # 6. create new model & load trained weights
    model = Yolonet(n_classes=len(config["model"]["labels"]))
    model.load_weights(os.path.join(config["train"]["save_folder"], "weights.h5"))
    detector = YoloDetector(model)
 
    # 7. predict & plot
    boxes = detector.detect(imgs[0], COCO_ANCHORS)
    image = draw_boxes(imgs[0], boxes, labels=config["model"]["labels"])
    plt.imshow(image)
    plt.show()


