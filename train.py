# -*- coding: utf-8 -*-

import tensorflow as tf
tf.enable_eager_execution()
import numpy as np

from yolo.yolonet import Yolonet
from yolo.loss import loss_fn


from yolo import PROJECT_ROOT
import os
TEST_SAMPLE_ROOT = os.path.join(PROJECT_ROOT, "tests", "samples")
def _list_y_trues():
    # (2, 9, 9, 3, 6) (2, 18, 18, 3, 6) (2, 36, 36, 3, 6)
    y_true_1 = np.load(os.path.join(TEST_SAMPLE_ROOT, "yolo_1.npy")).astype(np.float32)
    y_true_2 = np.load(os.path.join(TEST_SAMPLE_ROOT, "yolo_2.npy")).astype(np.float32)
    y_true_3 = np.load(os.path.join(TEST_SAMPLE_ROOT, "yolo_3.npy")).astype(np.float32)
    y_trues = [y_true_1, y_true_2, y_true_3]
    return y_trues

def _list_y_preds():
    # (2, 9, 9, 3, 6) (2, 18, 18, 3, 6) (2, 36, 36, 3, 6)
    y_pred_1 = np.load(os.path.join(TEST_SAMPLE_ROOT, "y_pred_1.npy")).astype(np.float32)
    y_pred_2 = np.load(os.path.join(TEST_SAMPLE_ROOT, "y_pred_2.npy")).astype(np.float32)
    y_pred_3 = np.load(os.path.join(TEST_SAMPLE_ROOT, "y_pred_3.npy")).astype(np.float32)
    ys_preds = [y_pred_1, y_pred_2, y_pred_3]
    return ys_preds

def _true_boxes():
    # (2, 1, 1, 1, 2, 4)
    true_boxes = np.load(os.path.join(TEST_SAMPLE_ROOT, "t_batch.npy")).astype(np.float32)
    return true_boxes

def _images():
    images = np.load(os.path.join(TEST_SAMPLE_ROOT, "x_batch.npy")).astype(np.float32)
    return images


# def grads_fn(self, input_data, target):
#     # 4) grads_fn(input_tensor, target_tensor)
#     # Get loss tensor

if __name__ == '__main__':
    
    images_tensor = tf.constant(_images())
    true_boxes = tf.constant(_true_boxes())
    list_y_trues = [tf.constant(arr) for arr in _list_y_trues()]

    yolo_model = Yolonet(n_features=18)    # 3 * (1+1+4)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    NUM_EPOCHES = 100
    for i in range(NUM_EPOCHES):

        # 1. prediction
        list_y_preds = yolo_model(images_tensor, training=True)

        # 2. loss
        with tf.GradientTape() as tape:
            loss_tenosr = loss_fn(true_boxes, list_y_trues, list_y_preds)
            # 3. grad
            grads = tape.gradient(loss_tenosr, yolo_model.variables)
            for grad in grads[:20]:
                print(grad)
        break

#         print(loss_tenosr.numpy())
#         print(len(yolo_model.variables), len(grads))
#         # 4. update variables using optimizer
#         optimizer.apply_gradients(zip(grads, yolo_model.variables))
        

    
    
    
    
