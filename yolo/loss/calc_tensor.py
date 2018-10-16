# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
# tf.enable_eager_execution()
from yolo.loss.utils import adjust_pred_tensor, adjust_true_tensor
from yolo.loss.utils import conf_delta_tensor, intersect_areas_tensor, reshape_y_pred_tensor, setup_env_tensor
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
        self.anchors = tf.constant(anchors, dtype='float', shape=[1,1,1,3,2])

        # 1. setup
        y_pred = reshape_y_pred_tensor(y_pred)
        object_mask, grid_factor, grid_h, grid_w = setup_env_tensor(y_true)
        net_factor  = tf.reshape(tf.cast(self.image_size, tf.float32), [1,1,1,1,2])

        # 2. Adjust prediction
        pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class = adjust_pred_tensor(y_pred, self.cell_grid, grid_h, grid_w)

        # 3. Adjust ground truth
        true_box_xy, true_box_wh, true_box_conf, true_box_class = adjust_true_tensor(y_true)

        # 4. conf_delta tensor
        intersect_areas, pred_areas, true_areas = intersect_areas_tensor(y_true,
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

def y_true_to_true_boxes(y_trues, anchors):

    def _batch(y_true, anchors):
        true_boxes = []
        n_rows, n_cols = y_true.shape[:2]
        for r in range(n_rows):
            for c in range(n_cols):
                for b in range(3):
                    if y_true[r, c, b, 4] != 0:
                        box = y_true[r, c, b, :4]
                        tw = box[2]
                        th = box[3]
                        pw = anchors[2*b]
                        ph = anchors[2*b + 1]
                        box_ = [box[0], box[1], int(pw * np.exp(tw)), int(ph * np.exp(th))]
                        true_boxes.append(box_)
        true_boxes = np.array(true_boxes)
        return true_boxes
    
    batch_size = y_trues.shape[0]
    true_boxes = np.zeros((batch_size, 1, 1, 1, 30, 4))
    for i in range(batch_size):
        idx = 0
        true_boxes_abatch = _batch(y_trues[i].numpy(), anchors)
        for b in true_boxes_abatch:
            true_boxes[i, 0, 0, 0, idx, :] = b
            idx += 1
    return true_boxes


if __name__ == '__main__':
    import numpy as np
    import os
    from yolo import PROJECT_ROOT
    tf.enable_eager_execution()
    def test():
        x_batch = np.load(os.path.join(PROJECT_ROOT, "x_batch.npy")).astype(np.float32)
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