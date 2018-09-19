# -*- coding: utf-8 -*-

import numpy as np


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


IDX_X = 0
IDX_Y = 1
IDX_W = 2
IDX_H = 3
IDX_OBJECTNESS = 4
IDX_CLASS_PROB = 5


def decode_netout(netout, anchors, obj_thresh, net_h, net_w, nb_box=3):
    """
    # Args
        netout : (n_rows, n_cols, 3, 4+1+n_classes)
        anchors
        
    """
    grid_h, grid_w = netout.shape[:2]
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))

    boxes = []

    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    
    # (13, 13, 3, 80) = (13, 13, 3, 1) * (13, 13, 3, 80)
    netout[..., IDX_CLASS_PROB:]  = netout[..., IDX_OBJECTNESS][..., np.newaxis] * netout[..., IDX_CLASS_PROB:]
    netout[..., IDX_CLASS_PROB:] *= netout[..., IDX_CLASS_PROB:] > obj_thresh

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # 4th element is objectness score
                objectness = netout[row, col, b, IDX_OBJECTNESS]
                
                if(objectness.all() <= obj_thresh): continue
                
                # first 4 elements are x, y, w, and h
                x, y, w, h = netout[row, col, b, :IDX_H+1]
    
                x = (col + x) / grid_w # center position, unit: image width
                y = (row + y) / grid_h # center position, unit: image height
                w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
                h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height  
                
                # last elements are class probabilities
                classes = netout[row, col, b, IDX_CLASS_PROB:]
                
                box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
    
                boxes.append(box)

    return boxes


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

        
