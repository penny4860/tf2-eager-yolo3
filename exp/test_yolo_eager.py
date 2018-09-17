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
    
#     yolo = get_yolo_keras()
#     f5, f4, f3 = yolo.predict(images)
#     np.save("f5", f5)
#     np.save("f4", f4)
#     np.save("f3", f3)
    
    import tensorflow as tf
    tf.enable_eager_execution()
    yolo = get_yolo_eager()
    f5, f4, f3 = yolo.predict(images)
    f5_keras = np.load("f5.npy")
    f4_keras = np.load("f4.npy")
    f3_keras = np.load("f3.npy")

    print(np.allclose(f5_keras, f5))
    print(np.allclose(f4_keras, f4))
    print(np.allclose(f3_keras, f3))





