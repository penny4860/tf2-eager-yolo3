# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np


def cell_grid(max_grid, batch_size=2):

    max_grid_h, max_grid_w = max_grid

    cell_x = np.arange(max_grid_w)
    cell_x = np.tile(cell_x, [max_grid_h])
    cell_x = np.reshape(cell_x, (1, max_grid_h, max_grid_w, 1, 1))
    cell_x = cell_x.astype(np.float32)
    
    cell_y = np.transpose(cell_x, (0,2,1,3,4))
    cell_grid = np.tile(np.concatenate([cell_x,cell_y],-1), [batch_size, 1, 1, 3, 1])
    return cell_grid


from yolo_ import YoloLayer
if __name__ == '__main__':
    
    BATCH_SIZE = 2
    MAX_GRID = [288, 288]
    
    cell_grid_tensor = YoloLayer(max_grid=MAX_GRID, batch_size=BATCH_SIZE).cell_grid
    
    with tf.Session() as sess:
        cell_grid_value = sess.run(cell_grid_tensor)
    cell_grid = cell_grid(MAX_GRID, batch_size=BATCH_SIZE)

    print(np.allclose(cell_grid_value, cell_grid))



