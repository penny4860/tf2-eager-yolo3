# -*- coding: utf-8 -*-

import tensorflow as tf
import os
from tqdm import tqdm

from yolo.loss import loss_fn


def train_fn(model, train_generator, valid_generator, learning_rate=1e-4, num_epoches=500, save_dname=None):
    
    save_fname = _setup(save_dname)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    
    history = []
    for i in range(num_epoches):

        # 1. update params
        _loop_train(model, optimizer, train_generator)
        
        # 2. monitor validation loss
        loss_value = _loop_validation(model, valid_generator)
        print("{}-th loss = {}".format(i, loss_value))

        # 3. update weights
        history.append(loss_value)
        if save_fname is not None and loss_value == min(history):
            print("    update weight {}".format(loss_value))
            model.save_weights("{}.h5".format(save_fname))
    
    return history


def _loop_train(model, optimizer, generator):
    # one epoch
    
    n_steps = generator.steps_per_epoch
    for _ in tqdm(range(n_steps)):
        xs, yolo_1, yolo_2, yolo_3 = generator.next_batch()
        ys = [yolo_1, yolo_2, yolo_3]

        grads = _grad_fn(model, xs, ys)
        optimizer.apply_gradients(zip(grads, model.variables))


def _loop_validation(model, generator):
    # one epoch
    n_steps = generator.steps_per_epoch
    loss_value = 0
    for _ in range(n_steps):
        xs, yolo_1, yolo_2, yolo_3 = generator.next_batch()
        ys = [yolo_1, yolo_2, yolo_3]
        ys_ = model(xs)
        loss_value += loss_fn(ys, ys_)
    loss_value /= generator.steps_per_epoch
    return loss_value


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
