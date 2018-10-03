# -*- coding: utf-8 -*-

import numpy as np
from yolo.loss.utils import adjust_pred, adjust_true, cell_grid, conf_delta_fn, intersect_areas_fn, reshape_y_pred, setup_env
from yolo.loss.utils import loss_class_fn, loss_conf_fn, loss_wh_fn, loss_xy_fn, wh_scale_fn

class LossCalculator(object):
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
        # make the model settings persistent
        self.ignore_thresh  = ignore_thresh
        self.anchors        = np.array(anchors).reshape([1,1,1,3,2]).astype(np.float32)
        self.grid_scale     = grid_scale
        self.obj_scale      = obj_scale
        self.noobj_scale    = noobj_scale
        self.xywh_scale     = xywh_scale
        self.class_scale    = class_scale
        self.image_size = image_size        # (h, w)-ordered

        # make a persistent mesh grid
        self.cell_grid = cell_grid(max_grid, batch_size)

    def run(self, true_boxes, y_true, y_pred):
        
        # 1. setup
        y_pred = reshape_y_pred(y_pred)
        object_mask, grid_factor, grid_h, grid_w = setup_env(y_true)

        net_factor = np.array(self.image_size, dtype=np.float32).reshape([1,1,1,1,2])

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
    from yolo import PROJECT_ROOT
    import os
    image_size = [288, 288]
    t_batch = np.load(os.path.join(PROJECT_ROOT, "samples", "t_batch.npy"))

    # (2, 9, 9, 3, 6) (2, 18, 18, 3, 6) (2, 36, 36, 3, 6)
    y_true_1 = np.load(os.path.join(PROJECT_ROOT, "samples", "yolo_1.npy"))
    y_true_2 = np.load(os.path.join(PROJECT_ROOT, "samples", "yolo_2.npy"))
    y_true_3 = np.load(os.path.join(PROJECT_ROOT, "samples", "yolo_3.npy"))
    ys_trues = [y_true_1, y_true_2, y_true_3]

    y_pred_1 = np.load(os.path.join(PROJECT_ROOT, "samples", "y_pred_1.npy")).astype(np.float64)
    y_pred_2 = np.load(os.path.join(PROJECT_ROOT, "samples", "y_pred_2.npy")).astype(np.float64)
    y_pred_3 = np.load(os.path.join(PROJECT_ROOT, "samples", "y_pred_3.npy")).astype(np.float64)
    ys_preds = [y_pred_1, y_pred_2, y_pred_3]

    anchorss=[[90, 95, 92, 154, 139, 281],
              [42, 44, 56, 51, 72, 66],
              [17, 18, 28, 24, 36, 34]]
    
    losses = []    
    for i in range(3):
        y_preds = ys_preds[i]
        ys = ys_trues[i]
        anchors = anchorss[i]
        
        loss_calculator = LossCalculator(anchors=anchors,
                                         max_grid=[288*(2**i),288*(2**i)],
                                         image_size=image_size)
        loss_value = loss_calculator.run(t_batch, ys, y_preds)
        losses.append(loss_value)    
    
    expected_losses = [np.array([0.56469357, 5.286211]).reshape(2,),
                       np.array([0.05866125, 4.778614]).reshape(2,),
                       np.array([0.54328686, 11.0839405]).reshape(2,)]

    for loss_value, expected_loss in zip(losses, expected_losses):
        if np.allclose(loss_value, expected_loss) == True:
            print("main : test passed")
        else:
            print("main : test failed")

# from yolo_ import YoloLayer
if __name__ == '__main__':
    test_main()

