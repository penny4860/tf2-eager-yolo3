# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from scipy.special import expit as sigmoid


def reshape_y_pred(y_pred):
    y_pred_reshaped = y_pred.reshape(y_pred.shape[0], y_pred.shape[1], y_pred.shape[2], 3, -1)
    return y_pred_reshaped

def setup_env(y_true):
    # initialize the masks
    object_mask     = np.expand_dims(y_true[..., 4], 4)
 
    # compute grid factor and net factor
    grid_h = y_true.shape[1]
    grid_w = y_true.shape[2]
    grid_factor = np.array([grid_w, grid_h], dtype=np.float32).reshape([1,1,1,1,2])
    return object_mask, grid_factor, grid_h, grid_w

def adjust_pred(y_pred, cell_grid, grid_h, grid_w):
    pred_box_xy    = (cell_grid[:,:grid_h,:grid_w,:,:] + sigmoid(y_pred[..., :2]))  # sigma(t_xy) + c_xy
    pred_box_wh    = y_pred[..., 2:4]                                                       # t_wh
    pred_box_conf  = np.expand_dims(sigmoid(y_pred[..., 4]), 4)                          # adjust confidence
    pred_box_class = y_pred[..., 5:]                                                        # adjust class probabilities      
    return pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class

def adjust_true(y_true):
    true_box_xy    = y_true[..., 0:2] # (sigma(t_xy) + c_xy)
    true_box_wh    = y_true[..., 2:4] # t_wh
    true_box_conf  = np.expand_dims(y_true[..., 4], 4)
    true_box_class = np.argmax(y_true[..., 5:], -1)
    return true_box_xy, true_box_wh, true_box_conf, true_box_class


def intersect_areas_fn(true_boxes, pred_box_xy, pred_box_wh, grid_factor, net_factor, anchors):
    """
    # Args
        true_boxes : (batch_size, 1, 1, 1, nb_box, 4)
        pred_box_xy : (batch_size, grid, grid, nb_box, 2)
        pred_box_wh : (batch_size, grid, grid, nb_box, 2)
        grid_factor : (1, 1, 1, 1, 2)
        net_factor : (1, 1, 1, 1, 2)
        anchors : (1, 1, 1, nb_box, 2)

    # Returns
        intersect_areas : (batch_size, grid, grid, nb_box, 2)
        pred_areas : (batch_size, grid, grid, nb_box, 1)
        true_areas : (batch_size, 1, 1, 1, 2)
    """
    # then, ignore the boxes which have good overlap with some true box
    true_xy = true_boxes[..., 0:2] / grid_factor
    true_wh = true_boxes[..., 2:4] / net_factor
     
    true_wh_half = true_wh / 2.
    true_mins    = true_xy - true_wh_half
    true_maxes   = true_xy + true_wh_half
     
    pred_xy = np.expand_dims(pred_box_xy / grid_factor, 4)
    pred_wh = np.expand_dims(np.exp(pred_box_wh) * anchors / net_factor, 4)
     
    pred_wh_half = pred_wh / 2.
    pred_mins    = pred_xy - pred_wh_half
    pred_maxes   = pred_xy + pred_wh_half    
    
    intersect_mins  = np.maximum(pred_mins,  true_mins)
    intersect_maxes = np.minimum(pred_maxes, true_maxes)
    
    intersect_wh    = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    return intersect_areas, pred_areas, true_areas


def conf_delta_fn(pred_box_conf, intersect_areas, pred_areas, true_areas, ignore_thresh):
    """
    # Args
        pred_box_conf
        intersect_areas : (batch_size, grid, grid, nb_box, 2)
        pred_areas : (batch_size, grid, grid, nb_box, 1)
        true_areas : (batch_size, 1, 1, 1, 2)
    # Returns
        conf_delta : (batch_size, grid, grid, nb_box, 1)
    """
    
    # initially, drag all objectness of all boxes to 0
    conf_delta  = pred_box_conf - 0 

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = intersect_areas / union_areas
    best_ious   = np.max(iou_scores, axis=4)        
    conf_delta *= np.expand_dims((best_ious < ignore_thresh).astype(np.float32), 4)
    return conf_delta

def wh_scale_fn(true_box_wh, anchors, net_factor):
    wh_scale = np.exp(true_box_wh) * anchors / net_factor
    wh_scale = np.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4) # the smaller the box, the bigger the scale
    return wh_scale

def loss_xy_fn(object_mask, pred_box_xy, true_box_xy, wh_scale, xywh_scale):
    xy_delta    = object_mask   * (pred_box_xy-true_box_xy) * wh_scale * xywh_scale
    loss_xy    = np.sum(xy_delta*xy_delta, tuple(range(1,5)))
    return loss_xy

def loss_wh_fn(object_mask, pred_box_wh, true_box_wh, wh_scale, xywh_scale):
    wh_delta    = object_mask   * (pred_box_wh-true_box_wh) * wh_scale * xywh_scale
    loss_wh    = np.sum(wh_delta*wh_delta,       tuple(range(1,5)))
    return loss_wh
    
def loss_conf_fn(object_mask, pred_box_conf, true_box_conf, obj_scale, noobj_scale, conf_delta):
    conf_delta  = object_mask * (pred_box_conf-true_box_conf) * obj_scale + (1-object_mask) * conf_delta * noobj_scale
    loss_conf  = np.sum(conf_delta*conf_delta,     tuple(range(1,5)))
    return loss_conf

def loss_class_fn(object_mask, pred_box_class, true_box_class, class_scale):
    # Todo : numpy 로 구현하자.
    def sparse_softmax_cross_entropy_with_logits(labels, logits):
        op = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = tf.constant(labels),
                                                            logits = tf.constant(logits))
        
        with tf.Session() as sess:
            return sess.run(op)

    class_delta = object_mask * \
                  np.expand_dims(sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class), 4) * \
                  class_scale
    loss_class = np.sum(class_delta,               tuple(range(1,5)))
    return loss_class

def cell_grid(max_grid, batch_size=2):

    max_grid_h, max_grid_w = max_grid
    
    cell_x = np.arange(max_grid_w)
    cell_x = np.tile(cell_x, [max_grid_h])
    cell_x = np.reshape(cell_x, (1, max_grid_h, max_grid_w, 1, 1))
    cell_x = cell_x.astype(np.float32)
    cell_y = np.transpose(cell_x, (0,2,1,3,4))
    cell_grid = np.tile(np.concatenate([cell_x,cell_y],-1), [batch_size, 1, 1, 3, 1])
    return cell_grid


class YoloLayerNp(object):
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

    def call(self, true_boxes, y_true, y_pred, image_h, image_w):
        
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

    yolo_layer_np = YoloLayerNp()
    loss_value = yolo_layer_np.call(t_batch, ys, y_preds, x_batch.shape[1], x_batch.shape[2])
    print(loss_value.shape)
    
    if np.allclose(loss_value, np.array([0.56469357, 5.286211]).reshape(2,)) == True:
        print("main : test passed")
    else:
        print("main : test failed")



# from yolo_ import YoloLayer
if __name__ == '__main__':
    test_main()

