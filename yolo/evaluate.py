# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
from tqdm import tqdm

from yolo.utils.box import draw_boxes
from yolo.dataset.annotation import parse_annotation
from yolo.eval.fscore import count_true_positives, calc_score



class Evaluator(object):
    def __init__(self, yolo_detector, class_labels, ann_fnames, img_dname):
        self._detector = yolo_detector
        self._cls_labels = class_labels
        self._ann_fnames = ann_fnames
        self._img_dname = img_dname
    
    def run(self, threshold=0.5, save_dname=None):
        n_true_positives = 0
        n_truth = 0
        n_pred = 0
        for ann_fname in tqdm(self._ann_fnames):
            img_fname, true_boxes, true_labels = parse_annotation(ann_fname, self._img_dname, self._cls_labels)
            true_labels = np.array(true_labels)
            image = cv2.imread(img_fname)[:,:,::-1]
    
            boxes, labels, probs = self._detector.detect(image, threshold)
            
            n_true_positives += count_true_positives(boxes, true_boxes, labels, true_labels)
            n_truth += len(true_boxes)
            n_pred += len(boxes)
            
            if save_dname:
                self._save_img(save_dname, img_fname, image, boxes, labels, probs)
        return calc_score(n_true_positives, n_truth, n_pred)

    def _save_img(self, save_dname, img_fname, image, boxes, labels, probs):
        if not os.path.exists(save_dname):
            os.makedirs(save_dname)
        image_ = draw_boxes(image, boxes, labels, probs, self._cls_labels, desired_size=416)
        output_path = os.path.join(save_dname, os.path.split(img_fname)[-1])
        cv2.imwrite(output_path, image_[:,:,::-1])
