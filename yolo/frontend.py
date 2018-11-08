# -*- coding: utf-8 -*-

import cv2
import os
import numpy as np
from tqdm import tqdm

from yolo.post_proc.decoder import postprocess_ouput
from yolo.net.yolonet import preprocess_input
from yolo.utils.box import boxes_to_array, to_minmax, draw_boxes
from yolo.dataset.annotation import parse_annotation
from yolo.eval.fscore import count_true_positives, calc_score


class YoloDetector(object):
    
    def __init__(self, model, anchors, net_size=288):
        self._model = model
        self._anchors = anchors
        self._net_size = net_size
        
    def detect(self, image, cls_threshold=0.0):
        """
        # Args
            image : array, shape of (H, W, 3)
            anchors : list, length of 18
            net_size : int
        # Returns
            boxes : array, shape of (N, 4)
                (x1, y1, x2, y2) ordered boxes
            labels : array, shape of (N,)
            probs : array, shape of (N,)
        """
        image_h, image_w, _ = image.shape
        new_image = preprocess_input(image, self._net_size)
        # 3. predict
        yolos = self._model.predict(new_image)
        boxes_ = postprocess_ouput(yolos, self._anchors, self._net_size, image_h, image_w)
        
        if len(boxes_) > 0:
            boxes, probs = boxes_to_array(boxes_)
            boxes = to_minmax(boxes)
            labels = np.array([b.get_label() for b in boxes_])
                        
            boxes = boxes[probs >= cls_threshold]
            labels = labels[probs >= cls_threshold]
            probs = probs[probs >= cls_threshold]
        else:
            boxes, labels, probs = [], [], []
        return boxes, labels, probs


class Evaluator(object):
    def __init__(self, yolo_detector):
        self._detector = yolo_detector
    
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


