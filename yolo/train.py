# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import os

from yolo.loss import loss_fn


def _loop_train(model, optimizer, iterator):
    # one epoch
    
    n_steps = iterator.steps_per_epoch
    for _ in range(n_steps):
        xs, yolo_1, yolo_2, yolo_3 = iterator.get_next()
        ys = [yolo_1, yolo_2, yolo_3]

        grads = _grad_fn(model, xs, ys)
        optimizer.apply_gradients(zip(grads, model.variables))


def _loop_validation(model, iterator):
    # one epoch
    n_steps = iterator.steps_per_epoch
    loss_value = 0
    for _ in range(n_steps):
        xs, yolo_1, yolo_2, yolo_3 = iterator.get_next()
        ys = [yolo_1, yolo_2, yolo_3]
        ys_ = model(xs)
        loss_value += loss_fn(ys, ys_)
    loss_value /= iterator.steps_per_epoch
    return loss_value


def train_fn(model, train_iterator, valid_iterator, learning_rate=1e-4, num_epoches=500, save_dname=None):
    
    save_fname = _setup(save_dname)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    

    min_loss_value = np.inf        
    history = []
    for i in range(num_epoches):

        # one epoch
        _loop_train(model, optimizer, train_iterator)
        
        # check validation error
        loss_value = _loop_validation(model, valid_iterator)
        
        history.append(loss_value)
        print("{}-th loss = {}".format(i, loss_value))
        
        if save_fname is not None and min_loss_value > loss_value:
            print("    update weight {}".format(loss_value))
            min_loss_value = loss_value
            model.save_weights("{}.h5".format(save_fname))
    
    return history


def _setup(save_dname):
    if save_dname:
        if not os.path.exists(save_dname):
            os.makedirs(save_dname)
        save_fname = os.path.join(save_dname, "weights")
    else:
        save_fname = None
    return save_fname


def _grad_fn(model, images_tensor, list_y_trues):
    with tf.GradientTape() as tape:
        logits = model(images_tensor)
        loss = loss_fn(list_y_trues, logits)
    return tape.gradient(loss, model.variables)


if __name__ == '__main__':
    tf.enable_eager_execution()
    pass
