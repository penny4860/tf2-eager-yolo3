# -*- coding: utf-8 -*-

from yolo.network import get_darknet
from yolo.darknet53 import Darknet53


def get_darknet_eager():
    from yolo import DARKNET_WEIGHTS
    from yolo.weights import WeightReader
    darknet_eager = Darknet53()
    reader = WeightReader(DARKNET_WEIGHTS)
    reader.load_weights(darknet_eager)
    return darknet_eager


if __name__ == '__main__':
    darknet = get_darknet()
    
    params = darknet.get_layer("conv_0").get_weights()
    weights = params[0]
    
    darknet_eager = get_darknet_eager()
    weights_eager = darknet_eager.get_variables(0, "kernel")[0]
    
    print(type(weights), type(weights_eager))
    import numpy as np
    print(np.allclose(weights, weights_eager))






