# -*- coding: utf-8 -*-

import numpy as np
from yolo.loss.utils import adjust_pred, adjust_true, cell_grid, conf_delta_fn, intersect_areas_fn, reshape_y_pred, setup_env
from yolo.loss.utils import loss_class_fn, loss_conf_fn, loss_wh_fn, loss_xy_fn, wh_scale_fn

class LossCalculator(object):
    def __init__(self,
                 anchors=[90, 95, 92, 154, 139, 281],
                 max_grid=[288, 288], 
                 batch_size=2,
                 ignore_thresh=0.5, 
                 grid_scale=1,
                 obj_scale=5,
                 noobj_scale=1,
                 xywh_scale=1,
                 class_scale=1):
        # make the model settings persistent
        self.ignore_thresh  = ignore_thresh
        self.anchors        = np.array(anchors).reshape([1,1,1,3,2]).astype(np.float32)
        self.grid_scale     = grid_scale
        self.obj_scale      = obj_scale
        self.noobj_scale    = noobj_scale
        self.xywh_scale     = xywh_scale
        self.class_scale    = class_scale        

        # make a persistent mesh grid
        self.cell_grid = cell_grid(max_grid, batch_size)

    def run(self, true_boxes, y_true, y_pred, image_h, image_w):
        
        # 1. setup
        y_pred = reshape_y_pred(y_pred)
        object_mask, grid_factor, grid_h, grid_w = setup_env(y_true)

        net_factor = np.array([image_h, image_w], dtype=np.float32).reshape([1,1,1,1,2])

        # 2. Adjust prediction
        pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class = adjust_pred(y_pred, self.cell_grid, grid_h, grid_w)

        # 3. Adjust ground truth
        true_box_xy, true_box_wh, true_box_conf, true_box_class = adjust_true(y_true)

        # 4. conf_delta tensor
        intersect_areas, pred_areas, true_areas = intersect_areas_fn(true_boxes,
                                                                     pred_box_xy,
                                                                     pred_box_wh,
                                                                     grid_factor,
                                                                     net_factor,
                                                                     self.anchors)

        conf_delta = conf_delta_fn(pred_box_conf, intersect_areas, pred_areas, true_areas, self.ignore_thresh)

        # 5. loss tensor
        wh_scale =  wh_scale_fn(true_box_wh, self.anchors, net_factor)

        loss_xy = loss_xy_fn(object_mask, pred_box_xy, true_box_xy, wh_scale, self.xywh_scale)
        loss_wh = loss_wh_fn(object_mask, pred_box_wh, true_box_wh, wh_scale, self.xywh_scale)
        loss_conf = loss_conf_fn(object_mask, pred_box_conf, true_box_conf, self.obj_scale, self.noobj_scale, conf_delta)
        loss_class = loss_class_fn(object_mask, pred_box_class, true_box_class, self.class_scale)
        loss = loss_xy + loss_wh + loss_conf + loss_class
        return loss*self.grid_scale


def test_main():
    x_batch, t_batch, ys, y_preds = np.load("x_batch.npy"), np.load("t_batch.npy"), np.load("ys.npy"), np.load("y_preds.npy")
    print(x_batch.shape, t_batch.shape, ys.shape, y_preds.shape)

    loss_calculator = LossCalculator()
    loss_value = loss_calculator.run(t_batch, ys, y_preds, x_batch.shape[1], x_batch.shape[2])
    print(loss_value.shape)
    
    if np.allclose(loss_value, np.array([0.56469357, 5.286211]).reshape(2,)) == True:
        print("main : test passed")
    else:
        print("main : test failed")



# from yolo_ import YoloLayer
if __name__ == '__main__':
    test_main()

