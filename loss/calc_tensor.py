# -*- coding: utf-8 -*-

import tensorflow as tf
from yolo.loss.utils import adjust_pred_tensor, adjust_true_tensor
from yolo.loss.utils import conf_delta_tensor, intersect_areas_tensor, reshape_y_pred_tensor, setup_env_tensor
from yolo.loss.utils import loss_class_tensor, loss_conf_tensor, loss_wh_tensor, loss_xy_tensor, wh_scale_tensor


class LossTensorCalculator(object):
    def __init__(self,
                 anchors=[90, 95, 92, 154, 139, 281],
                 max_grid=[288, 288], 
                 image_size=[288, 288], 
                 batch_size=2,
                 ignore_thresh=0.5, 
                 grid_scale=1,
                 obj_scale=5,
                 noobj_scale=1,
                 xywh_scale=1,
                 class_scale=1):
        self.ignore_thresh  = ignore_thresh
        self.anchors        = tf.constant(anchors, dtype='float', shape=[1,1,1,3,2])
        self.grid_scale     = grid_scale
        self.obj_scale      = obj_scale
        self.noobj_scale    = noobj_scale
        self.xywh_scale     = xywh_scale
        self.class_scale    = class_scale        
        self.image_size = image_size        # (h, w)-ordered

        # make a persistent mesh grid
        max_grid_h, max_grid_w = max_grid

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))
        self.cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [batch_size, 1, 1, 3, 1])

    def run(self, true_boxes, y_true, y_pred):
        # 1. setup
        y_pred = reshape_y_pred_tensor(y_pred)
        object_mask, grid_factor, grid_h, grid_w = setup_env_tensor(y_true)
        net_factor  = tf.reshape(tf.cast(self.image_size, tf.float32), [1,1,1,1,2])

        # 2. Adjust prediction
        pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class = adjust_pred_tensor(y_pred, self.cell_grid, grid_h, grid_w)

        # 3. Adjust ground truth
        true_box_xy, true_box_wh, true_box_conf, true_box_class = adjust_true_tensor(y_true)

        # 4. conf_delta tensor
        intersect_areas, pred_areas, true_areas = intersect_areas_tensor(true_boxes,
                                                                         pred_box_xy,
                                                                         pred_box_wh,
                                                                         grid_factor,
                                                                         net_factor,
                                                                         self.anchors)

        conf_delta = conf_delta_tensor(pred_box_conf, intersect_areas, pred_areas, true_areas, self.ignore_thresh)

        # 5. loss tensor
        wh_scale =  wh_scale_tensor(true_box_wh, self.anchors, net_factor)

        loss_xy = loss_xy_tensor(object_mask, pred_box_xy, true_box_xy, wh_scale, self.xywh_scale)
        loss_wh = loss_wh_tensor(object_mask, pred_box_wh, true_box_wh, wh_scale, self.xywh_scale)
        loss_conf = loss_conf_tensor(object_mask, pred_box_conf, true_box_conf, self.obj_scale, self.noobj_scale, conf_delta)
        loss_class = loss_class_tensor(object_mask, pred_box_class, true_box_class, self.class_scale)
        loss = loss_xy + loss_wh + loss_conf + loss_class
        return loss*self.grid_scale

