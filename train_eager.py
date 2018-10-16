# -*- coding: utf-8 -*-

import tensorflow as tf
tf.enable_eager_execution()
from yolo.net import Yolonet
from yolo.train import train

YOLOV3_WEIGHTS = "yolov3.weights"

if __name__ == '__main__':
    import os
    from yolo.dataset.generator import create_generator
    from yolo import PROJECT_ROOT

    # 1. create generator
    ann_dir = os.path.join(PROJECT_ROOT, "samples", "anns")
    img_dir = os.path.join(PROJECT_ROOT, "samples", "imgs")
    generator = create_generator(img_dir, ann_dir)
 
    # 2. create model
    model = Yolonet(n_classes=1)
    model.load_darknet_params(YOLOV3_WEIGHTS, True)
     
    # 3. define optimizer    
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
      
    # 4. training
    train(generator, optimizer, model, 100, 1)


