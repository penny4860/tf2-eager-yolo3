# -*- coding: utf-8 -*-

from yolo.post_proc.decoder import decode_netout
from yolo.post_proc.box import correct_yolo_boxes, do_nms, draw_boxes

import numpy as np

WEIGHTS_FNAME = "weights_67_0.38.h5"


def predict(image, model,
            labels = ["raccoon"],
            obj_thresh = 0.5,
            nms_thresh = 0.45,
            anchors=[17,18, 28,24, 36,34, 42,44, 56,51, 72,66, 90,95, 92,154, 139,281]):
    netout = model.predict(np.expand_dims(image, axis=0))
    net_h, net_w, _ = image.shape
    image_h, image_w, _ = image.shape
    
    anchors = np.array(anchors).reshape(3, 6)

    boxes = []
    for i in range(len(netout)):
        # decode the output of the network
        boxes += decode_netout(netout[i][0], anchors[3-(i+1)], obj_thresh, net_h, net_w)

    # correct the sizes of the bounding boxes
    correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

    # suppress non-maximal boxes
    do_nms(boxes, nms_thresh)
    
    image = draw_boxes(image, boxes, labels)
    return image

