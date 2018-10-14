# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
tf.enable_eager_execution()
from yolo.net import Yolonet
from yolo.predict import predict

WEIGHTS_FNAME = "weights.h5"


if __name__ == '__main__':
    import os
    from yolo import PROJECT_ROOT
    from yolo.dataset.generator import create_generator

    # 1. create generator
    ann_dir = os.path.join(PROJECT_ROOT, "samples", "anns")
    img_dir = os.path.join(PROJECT_ROOT, "samples", "imgs")
    generator = create_generator(img_dir, ann_dir)

    x_batch, t_batch, y1, y2, y3 = generator[0]

    input_image = x_batch[0]
    
    # 2. create model
    model = Yolonet(n_classes=1)
    model.load_weights(WEIGHTS_FNAME)
    output_image = predict(input_image, model)
    plt.imshow(output_image)
    plt.show()

