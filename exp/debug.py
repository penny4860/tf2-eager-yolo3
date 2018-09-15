# -*- coding: utf-8 -*-

from yolo.darknet53 import Darknet53
from keras.models import load_model


def get_darknet_eager():
    from yolo import DARKNET_WEIGHTS
    from yolo.weights import WeightReader
    darknet_eager = Darknet53()
    reader = WeightReader(DARKNET_WEIGHTS)
    reader.load_weights(darknet_eager)
    return darknet_eager

def get_darknet_keras():
    from yolo import PROJECT_ROOT
    import os
    model = load_model(os.path.join(PROJECT_ROOT, "darknet53_weights.h5"))
    return model

if __name__ == '__main__':
#     darknet = get_darknet_keras()
#     weights = darknet.layers[1].get_weights()[0]
#     import numpy as np
#     np.save("conv0", weights)
    
    import tensorflow as tf
    tf.enable_eager_execution()
    darknet_eager = get_darknet_eager()
    weights_eager = darknet_eager.get_variables(0, "kernel")[0]
    import numpy as np
    np.save("conv0_eager", weights_eager.numpy())

    conv0 = np.load("conv0.npy")
    conv0_eager = weights_eager.numpy()

    print(conv0[0,0,0,0:4])
    print(conv0_eager[0,0,0,0:4])

#     [ 0.03592291 -0.03602274 -0.03412793  0.03506648]
#     [ 0.03353848  0.00134301 -0.02568628 -0.26286137]



    





