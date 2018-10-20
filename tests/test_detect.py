# -*- coding: utf-8 -*-

import cv2
import os

from yolo.utils.utils import download_if_not_exists
from yolo.net.yolonet import Yolonet
from yolo import COCO_ANCHORS, PROJECT_ROOT
from yolo.frontend import YoloDetector


def test_detect(setup_tf_eager):
    # Download if not exits weight file
    weights_path = os.path.join(PROJECT_ROOT, "tests", "samples", "yolov3.weights")
    image_path   = os.path.join(PROJECT_ROOT, "tests", "samples", "sample.jpeg")
    
    download_if_not_exists(weights_path,
                           "https://pjreddie.com/media/files/yolov3.weights")

    # 1. create yolo model & load weights
    yolov3 = Yolonet()
    yolov3.load_darknet_params(weights_path)

    # 2. preprocess the image
    image = cv2.imread(image_path)
    image = image[:,:,::-1]

    d = YoloDetector(yolov3)
    boxes = d.detect(image, COCO_ANCHORS, net_size=416)
    assert len(boxes) == 31


import pytest
if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])
 


