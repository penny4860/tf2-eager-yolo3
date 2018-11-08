# -*- coding: utf-8 -*-

import cv2
import os

from yolo.net.yolonet import Yolonet
from yolo import COCO_ANCHORS, PROJECT_ROOT
from yolo.frontend import YoloDetector


def test_detect(setup_tf_eager, setup_darknet_weights):
    
    darknet_weights = setup_darknet_weights
    image_path   = os.path.join(PROJECT_ROOT, "tests", "samples", "sample.jpeg")

    # 1. create yolo model & load weights
    yolov3 = Yolonet()
    yolov3.load_darknet_params(darknet_weights)

    # 2. preprocess the image
    image = cv2.imread(image_path)
    image = image[:,:,::-1]

    d = YoloDetector(yolov3, COCO_ANCHORS, net_size=416)
    boxes, labels, probs = d.detect(image)
    assert len(boxes) == 31


import pytest
if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])
 


