# -*- coding: utf-8 -*-

import numpy as np
from yolo.post_proc.box import correct_yolo_boxes, do_nms

IDX_X = 0
IDX_Y = 1
IDX_W = 2
IDX_H = 3
IDX_OBJECTNESS = 4
IDX_CLASS_PROB = 5


def postprocess_ouput(yolos, anchors, net_h, net_w, image_h, image_w, obj_thresh=0.5, nms_thresh=0.5):
    """
    # Args
        yolos : list of arrays
            Yolonet outputs
    
    """
    boxes = []
    for i in range(len(yolos)):
        # decode the output of the network
        boxes += decode_netout(yolos[i][0], anchors[i], obj_thresh, net_h, net_w)

    # correct the sizes of the bounding boxes
    correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

    # suppress non-maximal boxes
    do_nms(boxes, nms_thresh)
    return boxes


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.objness = objness
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score


def decode_netout(netout, anchors, obj_thresh, net_h, net_w, nb_box=3):
    """
    # Args
        netout : (n_rows, n_cols, 3, 4+1+n_classes)
        anchors
        
    """
    n_rows, n_cols = netout.shape[:2]
    netout = netout.reshape((n_rows, n_cols, nb_box, -1))

    boxes = []
    for row in range(n_rows):
        for col in range(n_cols):
            for b in range(nb_box):
                # 1. decode
                x, y, w, h = _decode_coords(netout, row, col, b, anchors)
                objectness, classes = _activate_probs(netout[row, col, b, IDX_OBJECTNESS],
                                                      netout[row, col, b, IDX_CLASS_PROB:],
                                                      obj_thresh)

                # 2. scale normalize                
                x /= n_cols
                y /= n_rows
                w /= net_w
                h /= net_h
                
                if objectness > obj_thresh:
                    box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
                    boxes.append(box)

    return boxes


def _decode_coords(netout, row, col, b, anchors):
    x, y, w, h = netout[row, col, b, :IDX_H+1]

    x = col + _sigmoid(x)
    y = row + _sigmoid(y)
    w = anchors[2 * b + 0] * np.exp(w)
    h = anchors[2 * b + 1] * np.exp(h)

    return x, y, w, h


def _activate_probs(objectness, classes, obj_thresh=0.3):
    """
    # Args
        objectness : scalar
        classes : (n_classes, )
    
    # Returns
        objectness_prob : (n_rows, n_cols, n_box)
        classes_conditional_probs : (n_rows, n_cols, n_box, n_classes)
    """
    # 1. sigmoid activation
    objectness_prob = _sigmoid(objectness)
    classes_probs = _sigmoid(classes)
    # 2. conditional probability
    classes_conditional_probs = classes_probs * objectness_prob
    # 3. thresholding
    classes_conditional_probs *= objectness_prob > obj_thresh
    return objectness_prob, classes_conditional_probs
    
    
def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


if __name__ == '__main__':
    
#     0 (13, 13, 255) [116, 90, 156, 198, 373, 326]
#     1 (26, 26, 255) [30, 61, 62, 45, 59, 119]
#     2 (52, 52, 255) [10, 13, 16, 30, 33, 23]

    np.random.seed(0)
    netout = np.random.randn(13, 13, 255)
    anchors = [116, 90, 156, 198, 373, 326]
    boxes = decode_netout(netout, anchors, obj_thresh=0.5, net_h=416, net_w=416)
    
    import pickle
    with open('expected_boxes.pkl', 'rb') as f:
        expected_boxes = pickle.load(f)

    for box, expected_box in zip(boxes, expected_boxes):
        assert box.xmin == expected_box.xmin
        assert box.ymin == expected_box.ymin
        assert box.xmax == expected_box.xmax
        assert box.ymax == expected_box.ymax
        assert box.objness == expected_box.objness
        # assert box.classes == expected_box.classes

    print("passed")

        
