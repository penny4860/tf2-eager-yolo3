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

        boxes[i].x = int(boxes[i].x * image_w)
        boxes[i].w = int(boxes[i].w * image_w)
        boxes[i].y = int(boxes[i].y * image_h)
        boxes[i].h = int(boxes[i].h * image_h)

        
def draw_boxes(image, boxes, labels, obj_thresh=0.0):
    for box in boxes:
        label = np.argmax(box.classes)
        label_str = labels[label]
        if box.classes[label] > obj_thresh:
            print(label_str + ': ' + str(box.classes[label]*100) + '%')
                
            # Todo: check this code
            if image.dtype == np.uint8:
                image = image.astype(np.int32)
            x1, y1, x2, y2 = box.as_minmax().astype(np.int32)
            
            cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 3)
            cv2.putText(image, 
                        label_str + ' ' + str(box.get_score()), 
                        (x1, y1 - 13), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1e-3 * image.shape[0], 
                        (0,255,0), 2)
    return image      


