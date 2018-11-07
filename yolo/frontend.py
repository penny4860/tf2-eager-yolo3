# -*- coding: utf-8 -*-

from yolo.post_proc.decoder import postprocess_ouput
from yolo.net.yolonet import preprocess_input
from yolo.utils.box import boxes_to_array, to_minmax
import numpy as np

class YoloDetector(object):
    
    def __init__(self, model):
        self._model = model
        
    def detect(self, image, anchors, net_size=288, cls_threshold=0.0):
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
        new_image = preprocess_input(image, net_size)
        # 3. predict
        yolos = self._model.predict(new_image)
        boxes_ = postprocess_ouput(yolos, anchors, net_size, image_h, image_w)
        
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



