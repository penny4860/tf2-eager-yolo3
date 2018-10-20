# -*- coding: utf-8 -*-

import os
import tensorflow as tf

from yolo.dataset.generator import create_generator
from yolo.net.yolonet import Yolonet
from yolo.train import train


def test_train(setup_tf_eager, setup_darknet_weights, setup_train_dirs):
    
    img_dir, ann_dir = setup_train_dirs
    darknet_weights = setup_darknet_weights

    # 1. create generator
    generator = create_generator(img_dir, ann_dir,
                                 batch_size=2,
                                 labels_naming=["raccoon"],
                                 jitter=False)
 
    # 2. create model
    model = Yolonet(n_classes=1)
    model.load_darknet_params(darknet_weights, True)
     
    # 3. define optimizer    
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
      
    # 4. training
    loss_history = train(generator, optimizer, model, 2, 1)
    assert loss_history[0] > loss_history[1]


import pytest
if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])
 


