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




YOLOV3_WEIGHTS = "yolov3.weights"

if __name__ == '__main__':
    
    config = {
    "model" : {
        "anchors":              [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
        "labels":               ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
        "coord_scale" :         1.0,
        "class_scale" :         1.0,
        "object_scale" :         5.0,
        "no_object_scale" :     1.0
    },
    "pretrained" : {
        "keras_format":             "",
        "darknet_format":           "",
    },
    "train" : {
        "min_size":           416,
        "max_size":           416,
        "actual_epoch":         20,
        "train_image_folder":   "tests/dataset/svhn/imgs/",
        "train_annot_folder":   "tests/dataset/svhn/anns/",
        "train_times":          5,
        "valid_image_folder":   "tests/dataset/svhn/imgs/",
        "valid_annot_folder":   "tests/dataset/svhn/anns/",
        "valid_times":          1,
        "batch_size":           2,
        "learning_rate":        1e-4,
        "saved_folder":           "svhn",
        "jitter":                false,
        "first_trainable_layer": "input_1",
        "is_only_detect" :         true
    }
}

    # 1. create generator
    ann_dir = os.path.join(PROJECT_ROOT, "samples", "raccoon", "anns")
    img_dir = os.path.join(PROJECT_ROOT, "samples", "raccoon", "imgs")
    generator = create_generator(img_dir, ann_dir,
                                 batch_size=2,
                                 labels_naming=["raccoon"],
                                 anchors=COCO_ANCHORS)
 
    # 2. create model
    model = Yolonet(n_classes=1)
    model.load_darknet_params(YOLOV3_WEIGHTS, True)
     
    # 3. define optimizer    
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
      
    # 4. training
    train(generator, optimizer, model, 100, 1, fname="weights")

    # 5. prepare sample images
    img_fnames = [os.path.join(img_dir, "raccoon-1.jpg"), os.path.join(img_dir, "raccoon-12.jpg")]
    imgs = [cv2.imread(fname)[:,:,::-1] for fname in img_fnames]

    # 6. create new model & load trained weights
    model = Yolonet(n_classes=1)
    model.load_weights("weights.h5")
    detector = YoloDetector(model)

    # 7. predict & plot
    boxes = detector.detect(imgs[0], COCO_ANCHORS)
    image = draw_boxes(imgs[0], boxes, labels=["ani"], obj_thresh=0.0)
    plt.imshow(image)
    plt.show()


