# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from yolo.net import Yolonet
from yolo.loss import loss_fn


def train(generator, optimizer, model, num_epoches=500, verbose=10):
    def grads_fn(images_tensor, list_y_trues, true_boxes):
        with tf.GradientTape() as tape:
            logits = model(images_tensor)
            loss = loss_fn(true_boxes, list_y_trues, logits)
        return tape.gradient(loss, model.variables)

    for i in range(num_epoches):
        x_batch, t_batch, yolo_1, yolo_2, yolo_3 = generator[i][0]

        images_tensor = tf.constant(x_batch.astype(np.float32))
        list_y_trues = [tf.constant(yolo_1.astype(np.float32)),
                        tf.constant(yolo_2.astype(np.float32)),
                        tf.constant(yolo_3.astype(np.float32))]
        true_boxes = tf.constant(t_batch.astype(np.float32))
        
        grads = grads_fn(images_tensor, list_y_trues, true_boxes)
        optimizer.apply_gradients(zip(grads, model.variables))
        if i==0 or (i+1)%verbose==0:
            logits = model(images_tensor)
            print("{}-th loss = {}".format(i, loss_fn(true_boxes, list_y_trues, logits)))


if __name__ == '__main__':
    tf.enable_eager_execution()
    from yolo.samples import sample_images, sample_list_y_trues, sample_true_boxes
    
    # 1. setup dataset
    images_tensor = tf.constant(sample_images())
    true_boxes = tf.constant(sample_true_boxes())
    list_y_trues = [tf.constant(arr) for arr in sample_list_y_trues()]

    # 2. create model
    model = Yolonet(18)

    # 3. define optimizer    
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    
    # 4. training
    # train(images_tensor, list_y_trues, true_boxes, optimizer, model, 2, 1)

