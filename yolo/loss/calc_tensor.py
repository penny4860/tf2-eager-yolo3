# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
# tf.enable_eager_execution()
from yolo.loss.utils import adjust_pred_tensor, adjust_true_tensor
from yolo.loss.utils import conf_delta_tensor, reshape_y_pred_tensor, setup_env_tensor
from yolo.loss.utils import loss_class_tensor, loss_conf_tensor, loss_wh_tensor, loss_xy_tensor, wh_scale_tensor

def sum_loss(losses):
    return tf.sqrt(tf.reduce_sum(losses))

def loss_fn(list_y_trues, list_y_preds,
            anchors=[17,18, 28,24, 36,34, 42,44, 56,51, 72,66, 90,95, 92,154, 139,281],
            image_size=[288, 288], 
            ignore_thresh=0.5, 
            grid_scale=1,
            obj_scale=5,
            noobj_scale=1,
            xywh_scale=1,
            class_scale=1):
    
    calculator = LossTensorCalculator(image_size=image_size,
                                        ignore_thresh=ignore_thresh, 
                                        grid_scale=grid_scale,
                                        obj_scale=obj_scale,
                                        noobj_scale=noobj_scale,
                                        xywh_scale=xywh_scale,
                                        class_scale=class_scale)
    loss_yolo_1 = calculator.run(list_y_trues[0], list_y_preds[0], max_grid=[1*num for num in image_size], anchors=anchors[12:])
    loss_yolo_2 = calculator.run(list_y_trues[1], list_y_preds[1], max_grid=[2*num for num in image_size], anchors=anchors[6:12])
    loss_yolo_3 = calculator.run(list_y_trues[2], list_y_preds[2], max_grid=[4*num for num in image_size], anchors=anchors[:6])
    return sum_loss([loss_yolo_1, loss_yolo_2, loss_yolo_3])


class LossTensorCalculator(object):
    def __init__(self,
                 image_size=[288, 288], 
                 ignore_thresh=0.5, 
                 grid_scale=1,
                 obj_scale=5,
                 noobj_scale=1,
                 xywh_scale=1,
                 class_scale=1):
        self.ignore_thresh  = ignore_thresh
        self.grid_scale     = grid_scale
        self.obj_scale      = obj_scale
        self.noobj_scale    = noobj_scale
        self.xywh_scale     = xywh_scale
        self.class_scale    = class_scale        
        self.image_size = image_size        # (h, w)-ordered

    def run(self, y_true, y_pred, max_grid=[288, 288], anchors=[90, 95, 92, 154, 139, 281]):

        # make a persistent mesh grid
        batch_size = tf.shape(y_true)[0]
        max_grid_h, max_grid_w = max_grid
        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))
        self.cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [batch_size, 1, 1, 3, 1])

        # 1. setup
        y_pred = reshape_y_pred_tensor(y_pred)
        object_mask, grid_h, grid_w = setup_env_tensor(y_true)
        net_factor  = tf.reshape(tf.cast(self.image_size, tf.float32), [1,1,1,1,2])

        # 2. Adjust prediction
        pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class = adjust_pred_tensor(y_pred, self.cell_grid, grid_h, grid_w)

        # 3. Adjust ground truth
        true_box_xy, true_box_wh, true_box_conf, true_box_class = adjust_true_tensor(y_true)

        # 4. conf_delta tensor
        conf_delta = conf_delta_tensor(y_true,
                                         pred_box_xy,
                                         pred_box_wh,
                                         pred_box_conf,
                                         anchors,
                                         self.ignore_thresh)

        # 5. loss tensor
        wh_scale =  wh_scale_tensor(true_box_wh, anchors, net_factor)

        loss_xy = loss_xy_tensor(object_mask, pred_box_xy, true_box_xy, wh_scale, self.xywh_scale)
        loss_wh = loss_wh_tensor(object_mask, pred_box_wh, true_box_wh, wh_scale, self.xywh_scale)
        loss_conf = loss_conf_tensor(object_mask, pred_box_conf, true_box_conf, self.obj_scale, self.noobj_scale, conf_delta)
        loss_class = loss_class_tensor(object_mask, pred_box_class, true_box_class, self.class_scale)
        loss = loss_xy + loss_wh + loss_conf + loss_class
        return loss*self.grid_scale

if __name__ == '__main__':
    import os
    from yolo import PROJECT_ROOT
    tf.enable_eager_execution()
    def test():
        yolo_1 = np.load(os.path.join(PROJECT_ROOT, "yolo_1.npy")).astype(np.float32)
        pred_yolo_1 = np.load(os.path.join(PROJECT_ROOT, "pred_yolo_1.npy")).astype(np.float32)

        calculator = LossTensorCalculator()
        loss_tensor = calculator.run(tf.constant(yolo_1), pred_yolo_1)
        loss_value =loss_tensor.numpy()[0]
        
        if np.allclose(loss_value, 63.16674):
            print("Test Passed")
        else:
            print("Test Failed")
            print(loss_value)

    test()