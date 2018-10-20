# -*- coding: utf-8 -*-

import cv2
import os
import tensorflow as tf

from yolo.dataset.generator import create_generator
from yolo.utils.utils import download_if_not_exists
from yolo.net.yolonet import Yolonet
from yolo import COCO_ANCHORS, PROJECT_ROOT
from yolo.frontend import YoloDetector
from yolo.train import train


def test_train(setup_tf_eager):
    
    # Todo : conftest
    # Download if not exits weight file
    weights_path = os.path.join(PROJECT_ROOT, "tests", "samples", "yolov3.weights")
    download_if_not_exists(weights_path,
                           "https://pjreddie.com/media/files/yolov3.weights")

    # 1. create generator
    ann_dir = os.path.join(PROJECT_ROOT, "samples", "raccoon", "anns")
    img_dir = os.path.join(PROJECT_ROOT, "samples", "raccoon", "imgs")
    generator = create_generator(img_dir, ann_dir,
                                 batch_size=2,
                                 labels_naming=["raccoon"])
 
    # 2. create model
    model = Yolonet(n_classes=1)
    model.load_darknet_params(weights_path, True)
     
    # 3. define optimizer    
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
      
    # 4. training
    loss_history = train(generator, optimizer, model, 2, 1)
    assert loss_history[0] > loss_history[1]


import pytest
if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])
 


