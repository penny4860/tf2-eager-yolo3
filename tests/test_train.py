# -*- coding: utf-8 -*-

import pytest

from yolo.net.yolonet import Yolonet
from yolo.train import train_fn
from yolo.dataset.generator import BatchGenerator


@pytest.fixture(scope='function')
def setup_generator(setup_train_dirs):
    ann_fnames, image_root = setup_train_dirs
    generator = BatchGenerator(ann_fnames, image_root,
                                 batch_size=2,
                                 labels=["raccoon"],
                                 min_net_size=288,
                                 max_net_size=288,    
                                 jitter=False)
    return generator


def test_overfit(setup_tf_eager, setup_darknet_weights, setup_generator):
    
    # 1. create generator
    generator = setup_generator
 
    # 2. create model
    model = Yolonet(n_classes=1)
    model.load_darknet_params(setup_darknet_weights, True)
     
    # 3. training
    loss_history = train_fn(model,
                            generator,
                            valid_generator=None,
                            num_epoches=3)
    assert loss_history[0] > loss_history[-1]


def test_train(setup_tf_eager, setup_darknet_weights, setup_generator):

    # 1. create generator
    generator = setup_generator
 
    # 2. create model
    model = Yolonet(n_classes=1)
    model.load_darknet_params(setup_darknet_weights, True)
     
    # 3. training
    loss_history = train_fn(model,
                            generator,
                            valid_generator=setup_generator,
                            num_epoches=3)
    assert loss_history[0] > loss_history[-1]
    

if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])
 


