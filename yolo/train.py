# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from yolo.net import Yolonet
from yolo.loss import loss_fn


def train(generator, optimizer, model, save_dname, num_epoches=500, verbose=10):
    import os
    save_fname = os.path.join(save_dname, "weights")
    
    def grads_fn(images_tensor, list_y_trues):
        with tf.GradientTape() as tape:
            logits = model(images_tensor)
            loss = loss_fn(list_y_trues, logits)
        return tape.gradient(loss, model.variables)

    min_loss_value = np.inf        
    history = []
    for i in range(num_epoches):
        x_batch, yolo_1, yolo_2, yolo_3 = generator[i]

        images_tensor = tf.constant(x_batch.astype(np.float32))
        list_y_trues = [tf.constant(yolo_1.astype(np.float32)),
                        tf.constant(yolo_2.astype(np.float32)),
                        tf.constant(yolo_3.astype(np.float32))]
        
        grads = grads_fn(images_tensor, list_y_trues)
        optimizer.apply_gradients(zip(grads, model.variables))

        if i==0 or (i+1)%verbose==0:
            logits = model(images_tensor)
            loss_value = loss_fn(list_y_trues, logits)
            history.append(loss_value)
            print("{}-th loss = {}".format(i, loss_value))
            
            if min_loss_value > loss_value:
                print("    update weight {}".format(loss_value))
                min_loss_value = loss_value
                model.save_weights("{}.h5".format(save_fname))
    
    return history

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

