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
    x_batch, t_batch, yolo_1, yolo_2, yolo_3 = generator[0]
 
    # 2. create model
    model = Yolonet(n_classes=1)
    model.load_darknet_params(YOLOV3_WEIGHTS, True)
    
    pred_yolo_1, pred_yolo_2, pred_yolo_3 = model.predict(x_batch)
    print(x_batch.shape, t_batch.shape, yolo_1.shape, yolo_2.shape, yolo_3.shape)
    print(pred_yolo_1.shape, pred_yolo_2.shape, pred_yolo_3.shape)

    import numpy as np
    np.save("x_batch", x_batch)
    np.save("t_batch", t_batch)
    np.save("yolo_1", yolo_1)
    np.save("yolo_2", yolo_2)
    np.save("yolo_3", yolo_3)
    
    np.save("pred_yolo_1", pred_yolo_1)
    np.save("pred_yolo_2", pred_yolo_2)
    np.save("pred_yolo_3", pred_yolo_3)



