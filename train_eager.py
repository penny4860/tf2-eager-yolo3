# -*- coding: utf-8 -*-

import pickle
import tensorflow as tf
tf.enable_eager_execution()
from yolo.net import Yolonet
from yolo import YOLOV3_WEIGHTS
from yolo.train import train

def normalize(image):
    return image/255.

if __name__ == '__main__':
    import os
    from yolo.dataset.annotation import parse_annotation
    from yolo.dataset.generator import BatchGenerator
    from yolo import PROJECT_ROOT
    ann_dir = os.path.join(PROJECT_ROOT, "samples", "anns")
    img_dir = os.path.join(PROJECT_ROOT, "samples", "imgs")
    train_anns = parse_annotation(ann_dir,
                                  img_dir,
                                  labels_naming=["raccoon"])
    generator = BatchGenerator(train_anns,
                               batch_size=2,
                               anchors=[17,18, 28,24, 36,34, 42,44, 56,51, 72,66, 90,95, 92,154, 139,281],
                               min_net_size=288,
                               max_net_size=288,
                               shuffle=False,
                               norm=normalize,
                               labels=["raccoon"])
 
    # 2. create model
    model = Yolonet(n_classes=1)
    model.load_darknet_params(YOLOV3_WEIGHTS, True)
     
    # 3. define optimizer    
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
      
    # 4. training
    train(generator, optimizer, model, 100, 1)


