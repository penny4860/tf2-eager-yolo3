# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


def adjust_pred_tensor(y_pred):

    grid_offset = _create_mesh_xy(*y_pred.shape[:4])
    
    pred_xy    = grid_offset + tf.sigmoid(y_pred[..., :2])            # sigma(t_xy) + c_xy
    pred_wh    = y_pred[..., 2:4]                                                       # t_wh
    pred_conf  = tf.sigmoid(y_pred[..., 4])                          # adjust confidence
    pred_classes = y_pred[..., 5:]                                              
    
    preds = tf.concat([pred_xy, pred_wh, tf.expand_dims(pred_conf, axis=-1), pred_classes], axis=-1)
    return preds

def adjust_true_tensor(y_true):
    true_    = y_true[..., :5]
    true_class = tf.argmax(y_true[..., 5:], -1)
    trues = tf.concat([true_, tf.expand_dims(tf.cast(true_class, tf.float32), -1)], axis=-1)
    return trues

def conf_delta_tensor(y_true, y_pred, anchors, ignore_thresh):

    pred_box_xy, pred_box_wh, pred_box_conf = y_pred[..., :2], y_pred[..., 2:4], y_pred[..., 4]

    
    anchor_grid = _create_mesh_anchor(anchors, *y_pred.shape[:4])
    true_wh = y_true[:,:,:,:,2:4]
    true_wh = anchor_grid * tf.exp(true_wh)
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
    
    conf_delta = pred_box_conf * tf.compat.v1.to_float(best_ious < ignore_thresh)
    return conf_delta

def wh_scale_tensor(true_box_wh, anchors, image_size):
    
    image_size_  = tf.reshape(tf.cast(image_size, tf.float32), [1,1,1,1,2])
    anchors_ = tf.constant(anchors, dtype='float', shape=[1,1,1,3,2])
    
    # [0, 1]-scaled width/height
    wh_scale = tf.exp(true_box_wh) * anchors_ / image_size_
    # the smaller the box, the bigger the scale
    wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4) 
    return wh_scale

def loss_coord_tensor(object_mask, pred_box, true_box, wh_scale, xywh_scale):
    xy_delta    = object_mask   * (pred_box-true_box) * wh_scale * xywh_scale
    loss_xy    = tf.reduce_sum(tf.square(xy_delta),       list(range(1,5)))
    return loss_xy
    
def loss_conf_tensor(object_mask, pred_box_conf, true_box_conf, obj_scale, noobj_scale, conf_delta):
    object_mask_ = tf.squeeze(object_mask, axis=-1)
    conf_delta  = object_mask_ * (pred_box_conf-true_box_conf) * obj_scale + (1-object_mask_) * conf_delta * noobj_scale
    loss_conf  = tf.reduce_sum(tf.square(conf_delta),     list(range(1,4)))
    return loss_conf

def loss_class_tensor(object_mask, pred_box_class, true_box_class, class_scale):
    true_box_class_ = tf.cast(true_box_class, tf.int64)
    class_delta = object_mask * \
                  tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class_, logits=pred_box_class), 4) * \
                  class_scale
    loss_class = tf.reduce_sum(class_delta,               list(range(1,5)))
    return loss_class


def _create_mesh_xy(batch_size, grid_h, grid_w, n_box):
    """
    # Returns
        mesh_xy : Tensor, shape of (batch_size, grid_h, grid_w, n_box, 2)
            [..., 0] means "grid_w"
            [..., 1] means "grid_h"
    """
    mesh_x = tf.compat.v1.to_float(tf.reshape(tf.tile(tf.range(grid_w), [grid_h]), (1, grid_h, grid_w, 1, 1)))
    mesh_y = tf.transpose(mesh_x, (0,2,1,3,4))
    mesh_xy = tf.tile(tf.concat([mesh_x,mesh_y],-1), [batch_size, 1, 1, n_box, 1])
    return mesh_xy

def _create_mesh_anchor(anchors, batch_size, grid_h, grid_w, n_box):
    """
    # Returns
        mesh_xy : Tensor, shape of (batch_size, grid_h, grid_w, n_box, 2)
            [..., 0] means "anchor_w"
            [..., 1] means "anchor_h"
    """
    mesh_anchor = tf.tile(anchors, [batch_size*grid_h*grid_w])
    mesh_anchor = tf.reshape(mesh_anchor, [batch_size, grid_h, grid_w, n_box, 2])
    mesh_anchor = tf.cast(mesh_anchor, tf.float32)
    return mesh_anchor

