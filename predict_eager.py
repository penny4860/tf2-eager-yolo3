# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
tf.enable_eager_execution()
from yolo.utils.box import draw_boxes
from yolo.net.yolonet import Yolonet
from yolo import COCO_ANCHORS
from yolo.frontend import YoloDetector

WEIGHTS_FNAME = "weights.h5"


if __name__ == '__main__':
    import os
    from yolo import PROJECT_ROOT
    import cv2
    image_path = os.path.join(PROJECT_ROOT, "samples", "raccoon", "imgs", "raccoon-1.jpg")
    image_path = os.path.join(PROJECT_ROOT, "samples", "raccoon", "imgs", "raccoon-12.jpg")

    image = cv2.imread(image_path)
    image = image[:,:,::-1]

    # 2. create model
    model = Yolonet(n_classes=1)
    model.load_weights(WEIGHTS_FNAME)
    
    detector = YoloDetector(model)

    # 3. predict
    boxes = detector.detect(image, COCO_ANCHORS)
    
    # 4. draw detected boxes
    image = draw_boxes(image, boxes, labels=["ani"], obj_thresh=0.0)

    # 5. plot    
    plt.imshow(image)
    plt.show()





