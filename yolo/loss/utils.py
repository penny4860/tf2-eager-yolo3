# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


def adjust_pred_tensor(y_pred):
    
    # make a persistent mesh grid
    batch_size, grid_h, grid_w = tf.shape(y_pred)[0:3]
    cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(grid_w), [grid_h]), (1, grid_h, grid_w, 1, 1)))
    cell_y = tf.transpose(cell_x, (0,2,1,3,4))
    cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [batch_size, 1, 1, 3, 1])
    
    pred_box_xy    = cell_grid + tf.sigmoid(y_pred[..., :2])            # sigma(t_xy) + c_xy
    pred_box_wh    = y_pred[..., 2:4]                                                       # t_wh
    pred_box_conf  = tf.sigmoid(y_pred[..., 4])                          # adjust confidence
    pred_box_class = y_pred[..., 5:]                                                        # adjust class probabilities      
    return pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class

def adjust_true_tensor(y_true):
    true_box_xy    = y_true[..., 0:2] # (sigma(t_xy) + c_xy)
    true_box_wh    = y_true[..., 2:4] # t_wh
    true_box_conf  = y_true[..., 4]
    true_box_class = tf.argmax(y_true[..., 5:], -1)
    return true_box_xy, true_box_wh, true_box_conf, true_box_class

def conf_delta_tensor(y_true, pred_box_xy, pred_box_wh, pred_box_conf, anchors, ignore_thresh):
    batch_size, grid_size, _, n_box = y_true.shape[:4]
    cell_box = tf.tile(anchors, [batch_size*grid_size*grid_size])
    cell_box = tf.reshape(cell_box, [batch_size, grid_size, grid_size, n_box, 2])
    cell_box = tf.cast(cell_box, tf.float32)
    true_wh = y_true[:,:,:,:,2:4]
    true_wh = cell_box * tf.exp(true_wh)
    true_wh = true_wh * tf.expand_dims(y_true[:,:,:,:,4], 4)
    
    anchors_ = tf.constant(anchors, dtype='float', shape=[1,1,1,3,2])
    
    # then, ignore the boxes which have good overlap with some true box
    true_xy = y_true[..., 0:2]
     
    true_wh_half = true_wh / 2.
    true_mins    = true_xy - true_wh_half
    true_maxes   = true_xy + true_wh_half
     
    pred_xy = pred_box_xy
    pred_wh = tf.exp(pred_box_wh) * anchors_
     
    pred_wh_half = pred_wh / 2.
    pred_mins    = pred_xy - pred_wh_half
    pred_maxes   = pred_xy + pred_wh_half    
    
    intersect_mins  = tf.maximum(pred_mins,  true_mins)
    intersect_maxes = tf.minimum(pred_maxes, true_maxes)
    
    intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    best_ious  = tf.truediv(intersect_areas, union_areas)
    
    conf_delta = pred_box_conf * tf.to_float(best_ious < ignore_thresh)
    return conf_delta

def wh_scale_tensor(true_box_wh, anchors, image_size):
    
    image_size_  = tf.reshape(tf.cast(image_size, tf.float32), [1,1,1,1,2])
    anchors_ = tf.constant(anchors, dtype='float', shape=[1,1,1,3,2])
    
    # [0, 1]-scaled width/height
    wh_scale = tf.exp(true_box_wh) * anchors_ / image_size_
    # the smaller the box, the bigger the scale
    wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4) 
    return wh_scale

def loss_xy_tensor(object_mask, pred_box_xy, true_box_xy, wh_scale, xywh_scale):
    xy_delta    = object_mask   * (pred_box_xy-true_box_xy) * wh_scale * xywh_scale
    loss_xy    = tf.reduce_sum(tf.square(xy_delta),       list(range(1,5)))
    return loss_xy

def loss_wh_tensor(object_mask, pred_box_wh, true_box_wh, wh_scale, xywh_scale):
    wh_delta    = object_mask   * (pred_box_wh-true_box_wh) * wh_scale * xywh_scale
    loss_wh    = tf.reduce_sum(tf.square(wh_delta),       list(range(1,5)))
    return loss_wh
    
def loss_conf_tensor(object_mask, pred_box_conf, true_box_conf, obj_scale, noobj_scale, conf_delta):
    object_mask_ = tf.squeeze(object_mask, axis=-1)
    conf_delta  = object_mask_ * (pred_box_conf-true_box_conf) * obj_scale + (1-object_mask_) * conf_delta * noobj_scale
    loss_conf  = tf.reduce_sum(tf.square(conf_delta),     list(range(1,4)))
    return loss_conf

def loss_class_tensor(object_mask, pred_box_class, true_box_class, class_scale):
    class_delta = object_mask * \
                  tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class), 4) * \
                  class_scale
    loss_class = tf.reduce_sum(class_delta,               list(range(1,5)))
    return loss_class
