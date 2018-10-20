# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import os

from yolo.loss import loss_fn


def _setup(save_dname):
    if save_dname:
        if not os.path.exists(save_dname):
            os.makedirs(save_dname)
        save_fname = os.path.join(save_dname, "weights")
    else:
        save_fname = None
    return save_fname

def train(model, train_iterator, valid_iterator, learning_rate=1e-4, num_epoches=500, verbose=10, save_dname=None):
    
    save_fname = _setup(save_dname)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    def grads_fn(images_tensor, list_y_trues):
        with tf.GradientTape() as tape:
            logits = model(images_tensor)
            loss = loss_fn(list_y_trues, logits)
        return tape.gradient(loss, model.variables)

    min_loss_value = np.inf        
    history = []
    for i in range(num_epoches):
        xs, yolo_1, yolo_2, yolo_3 = train_iterator.get_next()
        ys = [yolo_1, yolo_2, yolo_3]

        grads = grads_fn(xs, ys)
        optimizer.apply_gradients(zip(grads, model.variables))

        if i==0 or (i+1)%verbose==0:
            xs, yolo_1, yolo_2, yolo_3 = valid_iterator.get_next()
            
            ys_ = model(xs)
            loss_value = loss_fn(ys, ys_)
            history.append(loss_value)
            print("{}-th loss = {}".format(i, loss_value))
            
            if save_fname is not None and min_loss_value > loss_value:
                print("    update weight {}".format(loss_value))
                min_loss_value = loss_value
                model.save_weights("{}.h5".format(save_fname))
    
    return history

if __name__ == '__main__':
    tf.enable_eager_execution()
    pass
