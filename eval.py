# -*- coding: utf-8 -*-
# Todo : eval.py 에서 config parser를 사용
import tensorflow as tf
tf.enable_eager_execution()

import cv2
import os
import argparse
import json
from tqdm import tqdm

from yolo.utils.box import draw_boxes

argparser = argparse.ArgumentParser(
    description='evaluate yolo-v3 network')

argparser.add_argument(
    '-c',
    '--config',
    default="configs/svhn.json",
    help='config file')

argparser.add_argument(
    '-s',
    '--save_dname',
    default=None)

argparser.add_argument(
    '-t',
    '--threshold',
    type=float,
    default=0.5)


import numpy as np
from yolo.dataset.annotation import parse_annotation
from yolo.eval.fscore import count_true_positives, calc_score
class Evaluator(object):
    def __init__(self, detector):
        self._detector = detector
    
    def run(self, ann_fnames, img_dname, class_labels, threshold=0.5, save_dname=None):
        n_true_positives = 0
        n_truth = 0
        n_pred = 0
        for ann_fname in tqdm(ann_fnames):
            img_fname, true_boxes, true_labels = parse_annotation(ann_fname, img_dname, class_labels)
            true_labels = np.array(true_labels)
            image = cv2.imread(img_fname)[:,:,::-1]
    
            boxes, labels, probs = self._detector.detect(image, threshold)
            
            n_true_positives += count_true_positives(boxes, true_boxes, labels, true_labels)
            n_truth += len(true_boxes)
            n_pred += len(boxes)
            
            if save_dname:
                if not os.path.exists(save_dname):
                    os.makedirs(save_dname)
                image_ = draw_boxes(image, boxes, labels, probs, class_labels, desired_size=416)
                output_path = os.path.join(save_dname, os.path.split(img_fname)[-1])
                cv2.imwrite(output_path, image_[:,:,::-1])
        return calc_score(n_true_positives, n_truth, n_pred)

if __name__ == '__main__':
    from yolo.config import ConfigParser
    args = argparser.parse_args()
    config_parser = ConfigParser(args.config)
    model = config_parser.create_model()
    detector = config_parser.create_detector(model)
 
    evaluator = Evaluator(detector)
    score = evaluator.run(config_parser.get_train_anns(),
                          config_parser._train_config["train_image_folder"],
                          config_parser.get_labels())
    
    print(score)

