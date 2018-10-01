# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


# def cell_grid():
#     # make a persistent mesh grid
#     max_grid_h, max_grid_w = max_grid


def cell_grid_tf(max_grid, batch_size=2):
    max_grid_h, max_grid_w = max_grid

    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)))
    cell_y = tf.transpose(cell_x, (0,2,1,3,4))
    cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [batch_size, 1, 1, 3, 1])
    return cell_grid


from yolo_ import YoloLayer
if __name__ == '__main__':
    cell_grid_tensor = cell_grid_tf([10, 10])
    
    with tf.Session() as sess:
        cell_grid = sess.run(cell_grid_tensor)
        print(cell_grid.shape)



