# -*- coding: utf-8 -*-

import tensorflow as tf
tf.enable_eager_execution()

from yolo.yolonet import Yolonet
from yolo.loss import loss_fn


def train(images_tensor, list_y_trues, true_boxes, optimizer, model, num_epoches=500, verbose=10):
    def grads_fn(images_tensor, list_y_trues, true_boxes):
        with tf.GradientTape() as tape:
            logits = model(images_tensor)
            loss = loss_fn(true_boxes, list_y_trues, logits)
        return tape.gradient(loss, model.variables)

    for i in range(num_epoches):
        
        grads = grads_fn(images_tensor, list_y_trues, true_boxes)
        optimizer.apply_gradients(zip(grads, model.variables))
        if i==0 or (i+1)%verbose==0:
            logits = model(images_tensor)
            print("{}-th loss = {}".format(i, loss_fn(true_boxes, list_y_trues, logits)))



if __name__ == '__main__':
    
    from yolo import PROJECT_ROOT
    import os
    import numpy as np
    TEST_SAMPLE_ROOT = os.path.join(PROJECT_ROOT, "tests", "samples")
    def _list_y_trues():
        # (2, 9, 9, 3, 6) (2, 18, 18, 3, 6) (2, 36, 36, 3, 6)
        y_true_1 = np.load(os.path.join(TEST_SAMPLE_ROOT, "yolo_1.npy")).astype(np.float32)
        y_true_2 = np.load(os.path.join(TEST_SAMPLE_ROOT, "yolo_2.npy")).astype(np.float32)
        y_true_3 = np.load(os.path.join(TEST_SAMPLE_ROOT, "yolo_3.npy")).astype(np.float32)
        y_trues = [y_true_1, y_true_2, y_true_3]
        return y_trues
    
    def _true_boxes():
        # (2, 1, 1, 1, 2, 4)
        true_boxes = np.load(os.path.join(TEST_SAMPLE_ROOT, "t_batch.npy")).astype(np.float32)
        return true_boxes
    
    def _images():
        images = np.load(os.path.join(TEST_SAMPLE_ROOT, "x_batch.npy")).astype(np.float32)
        return images
    
    # 1. setup dataset
    images_tensor = tf.constant(_images())
    true_boxes = tf.constant(_true_boxes())
    list_y_trues = [tf.constant(arr) for arr in _list_y_trues()]

    # 2. create model
    model = Yolonet(18)

    # 3. define optimizer    
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    
    # 4. training
    train(images_tensor, list_y_trues, true_boxes, optimizer, model, 10, 1)

    
    
    
    
