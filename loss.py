# -*- coding: utf-8 -*-

import numpy as np
import os

from yolo_ import create_yolov3_model, create_loss_model
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
    
    # Todo : implement loss_fn()
    # loss_value = loss_fn(y_true, y_pred)


#     # should (loss_values == losses)
#     losses = train_model.predict([x_batch, t_batch, y_true_1, y_true_2, y_true_3])
