# -*- coding: utf-8 -*-

import numpy as np
import cv2

def correct_yolo_boxes(boxes, image_h, image_w):
    """
    # Args
        boxes : array, shape of (N, 4)
            [0, 1]-scaled box
    # Returns
        boxes : array shape of (N, 4)
            ([0, image_h], [0, image_w]) - scaled box
    """
    for i in range(len(boxes)):
        boxes[i].xmin = int(boxes[i].xmin * image_w)
        boxes[i].xmax = int(boxes[i].xmax * image_w)
        boxes[i].ymin = int(boxes[i].ymin * image_h)
        boxes[i].ymax = int(boxes[i].ymax * image_h)

        boxes[i].xmin = max(boxes[i].xmin, 0)
        boxes[i].xmax = min(boxes[i].xmax, image_w-1)
        boxes[i].ymin = max(boxes[i].ymin, 0)
        boxes[i].ymax = min(boxes[i].ymax, image_h-1)


        
def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
        
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if _bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0


def draw_boxes(image, boxes, labels, obj_thresh=0.0):
    for box in boxes:
        label = np.argmax(box.classes)
        label_str = labels[label]
        if box.classes[label] > obj_thresh:
            print(label_str + ': ' + str(box.classes[label]*100) + '%')
                
            # Todo: check this code
            if image.dtype == np.uint8:
                image = image.astype(np.int32)
            
            cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), (0,255,0), 3)
            cv2.putText(image, 
                        label_str + ' ' + str(box.get_score()), 
                        (box.xmin, box.ymin - 13), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1e-3 * image.shape[0], 
                        (0,255,0), 2)
    return image      


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3          

def _bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union


