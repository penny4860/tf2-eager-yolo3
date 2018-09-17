# -*- coding: utf-8 -*-

from yolo.yolonet import Yolonet
from yolo.network import make_yolov3_model

from yolo import YOLOV3_WEIGHTS


def get_yolo_eager():
    from yolo.weights import WeightReader
    yolo_eager = Yolonet()
    reader = WeightReader(YOLOV3_WEIGHTS)
    reader.load_weights(yolo_eager)
    return yolo_eager


def get_yolo_keras():
    from yolo.network import WeightReader
    yolo_keras = make_yolov3_model()
    reader = WeightReader(YOLOV3_WEIGHTS)
    reader.load_weights(yolo_keras)
    return yolo_keras


if __name__ == '__main__':
    import numpy as np
    np.random.seed(0)
    images = np.random.randn(1, 224, 224, 3)
    
    yolo = get_yolo_keras()
    f5, f4, f3 = yolo.predict(images)
    np.save("f5", f5)
    np.save("f4", f4)
    np.save("f3", f3)
    
    

#     darknet = get_darknet_keras()
#     ys = darknet.predict(images)
#     print(ys.shape)
#     np.save("ys_keras", ys)

#     import tensorflow as tf
#     tf.enable_eager_execution()
#     darknet_eager = get_darknet_eager()
#     ys = darknet_eager(tf.constant(images.astype(np.float32)))
#     ys_eager = ys.numpy()
#     ys_keras = np.load("ys_keras.npy")
#     print(np.allclose(ys_eager, ys_keras))





