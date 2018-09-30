# -*- coding: utf-8 -*-

import numpy as np
import os
from keras.layers import Input
from keras.models import Model

from yolo_ import create_yolov3_model, YoloLayer
from yolo import PROJECT_ROOT

def load_train_model_inputs():
    
    # (2, 288, 288, 3) (2, 1, 1, 1, 2, 4)
    x_batch = np.load(os.path.join(PROJECT_ROOT, "samples", "x_batch.npy"))
    t_batch = np.load(os.path.join(PROJECT_ROOT, "samples", "t_batch.npy"))

    # (2, 9, 9, 3, 6) (2, 18, 18, 3, 6) (2, 36, 36, 3, 6)
    yolo_1 = np.load(os.path.join(PROJECT_ROOT, "samples", "yolo_1.npy"))
    yolo_2 = np.load(os.path.join(PROJECT_ROOT, "samples", "yolo_2.npy"))
    yolo_3 = np.load(os.path.join(PROJECT_ROOT, "samples", "yolo_3.npy"))
    
    return x_batch, t_batch, yolo_1, yolo_2, yolo_3

def setup_graph_models():
    train_model, infer_model = create_yolov3_model(nb_class=1)
    infer_model.load_weights(os.path.join(PROJECT_ROOT, "raccoon.h5"))
    train_model.load_weights(os.path.join(PROJECT_ROOT, "raccoon.h5"))
    return train_model, infer_model


def create_loss_model(scale=1):
    def _yolo_layer(scale=1):
        if scale == 1:
            yolo_layer = YoloLayer(anchors = [90, 95, 92, 154, 139, 281],
                                   max_grid = [288, 288])
        if scale == 2:
            yolo_layer = YoloLayer(anchors = [42, 44, 56, 51, 72, 66],
                                   max_grid = [576, 576])
        if scale == 3:
            yolo_layer = YoloLayer(anchors = [17, 18, 28, 24, 36, 34],
                                   max_grid = [1152, 1152])
        return yolo_layer
    
    # 1. input tensors
    input_image = Input(shape=(None, None, 3)) # net_h, net_w, 3
    true_boxes  = Input(shape=(1, 1, 1, 2, 4))
    true_y = Input(shape=(None, None, 3, 4+1+1)) # grid_h, grid_w, nb_anchor, 5+nb_class
    pred_y = Input(shape=(None, None, 3*(4+1+1))) # grid_h, grid_w, nb_anchor, 5+nb_class
    
    # 2. create loss layer
    loss_layer = _yolo_layer(scale)
    output_tensor = loss_layer([input_image, pred_y, true_y, true_boxes])

    # 3. create loss model
    loss_model = Model([input_image, true_boxes, true_y, pred_y],
                       output_tensor)
    return loss_model

if __name__ == '__main__':
    
    train_model, infer_model = setup_graph_models()
    x_batch, t_batch, y_true_1, y_true_2, y_true_3 = load_train_model_inputs()
    y_pred_1, y_pred_2, y_pred_3 = infer_model.predict(x_batch)
    
    y_preds = [y_pred_1, y_pred_2, y_pred_3]
    ys = [y_true_1, y_true_2, y_true_3]
    scales = [1, 2, 3]
    
    for i in range(3):
        loss_model = create_loss_model(scales[i])
        loss_value = loss_model.predict([x_batch, t_batch, ys[i], y_preds[i]])
        print("scale: {}, loss_value: {}".format(i+1, loss_value))
        # scale: 1, loss_value: [0.56469357 5.286211  ]
        # scale: 2, loss_value: [0.05866125 4.778614  ]
        # scale: 3, loss_value: [ 0.54328686 11.0839405 ]

#     # should (loss_values == losses)
#     loss_values = loss_fn(y_true_1, y_true_2, y_true_3, y_pred_1, y_pred_2, y_pred_3)
#     losses = train_model.predict([x_batch, t_batch, y_true_1, y_true_2, y_true_3])
