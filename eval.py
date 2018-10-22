# -*- coding: utf-8 -*-

import tensorflow as tf
tf.enable_eager_execution()

import matplotlib.pyplot as plt
import cv2
import os
import argparse
import json
import glob

from yolo.net import Yolonet
from yolo.frontend import YoloDetector
from yolo.utils.box import boxes_to_array, to_minmax, draw_boxes

argparser = argparse.ArgumentParser(
    description='train yolo-v3 network')

argparser.add_argument(
    '-c',
    '--config',
    default="configs/raccoon.json",
    help='config file')


if __name__ == '__main__':
    from yolo.dataset.annotation import parse_annotation
    from yolo.eval.fscore import count_true_positives, calc_score
    import numpy as np
    
    args = argparser.parse_args()
    with open(args.config) as data_file:    
        config = json.load(data_file)

    model = Yolonet(n_classes=len(config["model"]["labels"]))
    model.load_weights(os.path.join(config["train"]["save_folder"], "weights.h5"))
    detector = YoloDetector(model)
 
    n_true_positives = 0
    n_truth = 0
    n_pred = 0
    ann_fnames = glob.glob(os.path.join(config["train"]["train_annot_folder"], "*.xml"))
    for ann_fname in ann_fnames: 
        img_fname, true_boxes, true_labels = parse_annotation(ann_fname, config["train"]["train_image_folder"], config["model"]["labels"])
        true_labels = np.array(true_labels)
        image = cv2.imread(img_fname)[:,:,::-1]

        boxes, labels, probs = detector.detect(image, config["model"]["anchors"])
        
        n_true_positives += count_true_positives(boxes, true_boxes, labels, true_labels)
        n_truth += len(true_boxes)
        n_pred += len(boxes)

        print(n_true_positives, n_truth, n_pred, probs)
        image = draw_boxes(image, boxes, labels, probs, class_labels=config["model"]["labels"])
        plt.imshow(image)
        plt.show()
    
    print(n_true_positives, n_truth, n_pred)
    print(calc_score(n_true_positives, n_truth, n_pred))


# Todo 1: detector.detect 의 리턴 타입을 np array 로 
# Todo 2: draw_boxes() 의 입력 타입을 np array 로 




