# -*- coding: utf-8 -*-

import numpy as np
import os
from keras.layers import Input
from keras.models import Model

from yolo_ import create_yolov3_model, YoloLayer
from yolo import PROJECT_ROOT

def load_train_model_inputs():
    
    # (2, 288, 288, 3) (2, 1, 1, 1, 2, 4)
    x_batch = np.load(os.path.join(PROJECT_ROOT, "samples", "x_batch.npy"))
    t_batch = np.load(os.path.join(PROJECT_ROOT, "samples", "t_batch.npy"))

    # (2, 9, 9, 3, 6) (2, 18, 18, 3, 6) (2, 36, 36, 3, 6)
    yolo_1 = np.load(os.path.join(PROJECT_ROOT, "samples", "yolo_1.npy"))
    yolo_2 = np.load(os.path.join(PROJECT_ROOT, "samples", "yolo_2.npy"))
    yolo_3 = np.load(os.path.join(PROJECT_ROOT, "samples", "yolo_3.npy"))
    
    return x_batch, t_batch, yolo_1, yolo_2, yolo_3

def setup_graph_models():
    train_model, infer_model = create_yolov3_model(nb_class=1)
    infer_model.load_weights(os.path.join(PROJECT_ROOT, "raccoon.h5"))
    train_model.load_weights(os.path.join(PROJECT_ROOT, "raccoon.h5"))
    return train_model, infer_model


def create_loss_model(scale=1):
    def _yolo_layer(scale=1):
        if scale == 1:
            yolo_layer = YoloLayer(anchors = [90, 95, 92, 154, 139, 281],
                                   max_grid = [288, 288])
        if scale == 2:
            yolo_layer = YoloLayer(anchors = [42, 44, 56, 51, 72, 66],
                                   max_grid = [576, 576])
        if scale == 3:
            yolo_layer = YoloLayer(anchors = [17, 18, 28, 24, 36, 34],
                                   max_grid = [1152, 1152])
        return yolo_layer
    
    # 1. input tensors
    input_image = Input(shape=(None, None, 3)) # net_h, net_w, 3
    true_boxes  = Input(shape=(1, 1, 1, 2, 4))
    true_y = Input(shape=(None, None, 3, 4+1+1)) # grid_h, grid_w, nb_anchor, 5+nb_class
    pred_y = Input(shape=(None, None, 3*(4+1+1))) # grid_h, grid_w, nb_anchor, 5+nb_class
    
    # 2. create loss layer
    loss_layer = _yolo_layer(scale)
    output_tensor = loss_layer([input_image, pred_y, true_y, true_boxes])

    # 3. create loss model
    loss_model = Model([input_image, true_boxes, true_y, pred_y],
                       output_tensor)
    return loss_model


def loss_fn(y_true, y_pred, scale=1):
    print(y_true.shape, y_pred.shape)


class YoloLayer(Layer):
    def __init__(self,
                 anchors=[90, 95, 92, 154, 139, 281],
                 max_grid=[288, 288], 
                 batch_size=2,
                 warmup_batches=0,
                 ignore_thresh=0.5, 
                 grid_scale=1,
                 obj_scale=5,
                 noobj_scale=1,
                 xywh_scale=1,
                 class_scale=1, 
                 **kwargs):
        # make the model settings persistent
        self.ignore_thresh  = ignore_thresh
        self.warmup_batches = warmup_batches
        self.anchors        = tf.constant(anchors, dtype='float', shape=[1,1,1,3,2])
        self.grid_scale     = grid_scale
        self.obj_scale      = obj_scale
        self.noobj_scale    = noobj_scale
        self.xywh_scale     = xywh_scale
        self.class_scale    = class_scale        

        # make a persistent mesh grid
        max_grid_h, max_grid_w = max_grid

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))
        self.cell_grid = tf.tile(tf.concat([cell_x,cell_y],-1), [batch_size, 1, 1, 3, 1])

        super(YoloLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(YoloLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        input_image, y_pred, y_true, true_boxes = x

        # adjust the shape of the y_predict [batch, grid_h, grid_w, 3, 4+1+nb_class]
        y_pred = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))
        
        # initialize the masks
        object_mask     = tf.expand_dims(y_true[..., 4], 4)

        # the variable to keep track of number of batches processed
        batch_seen = tf.Variable(0.)        

        # compute grid factor and net factor
        grid_h      = tf.shape(y_true)[1]
        grid_w      = tf.shape(y_true)[2]
        grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1,1,1,1,2])

        net_h       = tf.shape(input_image)[1]
        net_w       = tf.shape(input_image)[2]            
        net_factor  = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1,1,1,1,2])
        
        """
        Adjust prediction
        """
        pred_box_xy    = (self.cell_grid[:,:grid_h,:grid_w,:,:] + tf.sigmoid(y_pred[..., :2]))  # sigma(t_xy) + c_xy
        pred_box_wh    = y_pred[..., 2:4]                                                       # t_wh
        pred_box_conf  = tf.expand_dims(tf.sigmoid(y_pred[..., 4]), 4)                          # adjust confidence
        pred_box_class = y_pred[..., 5:]                                                        # adjust class probabilities      

        """
        Adjust ground truth
        """
        true_box_xy    = y_true[..., 0:2] # (sigma(t_xy) + c_xy)
        true_box_wh    = y_true[..., 2:4] # t_wh
        true_box_conf  = tf.expand_dims(y_true[..., 4], 4)
        true_box_class = tf.argmax(y_true[..., 5:], -1)         

        """
        Compare each predicted box to all true boxes
        """        
        # initially, drag all objectness of all boxes to 0
        conf_delta  = pred_box_conf - 0 

        # then, ignore the boxes which have good overlap with some true box
        true_xy = true_boxes[..., 0:2] / grid_factor
        true_wh = true_boxes[..., 2:4] / net_factor
        
        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half
        
        pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)
        pred_wh = tf.expand_dims(tf.exp(pred_box_wh) * self.anchors / net_factor, 4)
        
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
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        best_ious   = tf.reduce_max(iou_scores, axis=4)        
        conf_delta *= tf.expand_dims(tf.to_float(best_ious < self.ignore_thresh), 4)

        """
        Compute some online statistics
        """            
        true_xy = true_box_xy / grid_factor
        true_wh = tf.exp(true_box_wh) * self.anchors / net_factor

        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half

        pred_xy = pred_box_xy / grid_factor
        pred_wh = tf.exp(pred_box_wh) * self.anchors / net_factor 
        
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
        iou_scores  = tf.truediv(intersect_areas, union_areas)
        iou_scores  = object_mask * tf.expand_dims(iou_scores, 4)
        
        """
        Warm-up training
        """
        batch_seen = tf.assign_add(batch_seen, 1.)
        
        true_box_xy, true_box_wh, xywh_mask = tf.cond(tf.less(batch_seen, self.warmup_batches+1), 
                              lambda: [true_box_xy + (0.5 + self.cell_grid[:,:grid_h,:grid_w,:,:]) * (1-object_mask), 
                                       true_box_wh + tf.zeros_like(true_box_wh) * (1-object_mask), 
                                       tf.ones_like(object_mask)],
                              lambda: [true_box_xy, 
                                       true_box_wh,
                                       object_mask])

        """
        Compare each true box to all anchor boxes
        """      
        wh_scale = tf.exp(true_box_wh) * self.anchors / net_factor
        wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4) # the smaller the box, the bigger the scale

        xy_delta    = xywh_mask   * (pred_box_xy-true_box_xy) * wh_scale * self.xywh_scale
        wh_delta    = xywh_mask   * (pred_box_wh-true_box_wh) * wh_scale * self.xywh_scale
        conf_delta  = object_mask * (pred_box_conf-true_box_conf) * self.obj_scale + (1-object_mask) * conf_delta * self.noobj_scale
        class_delta = object_mask * \
                      tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class), 4) * \
                      self.class_scale

        loss_xy    = tf.reduce_sum(tf.square(xy_delta),       list(range(1,5)))
        loss_wh    = tf.reduce_sum(tf.square(wh_delta),       list(range(1,5)))
        loss_conf  = tf.reduce_sum(tf.square(conf_delta),     list(range(1,5)))
        loss_class = tf.reduce_sum(class_delta,               list(range(1,5)))

        loss = loss_xy + loss_wh + loss_conf + loss_class
        return loss*self.grid_scale


if __name__ == '__main__':
    
    train_model, infer_model = setup_graph_models()
    x_batch, t_batch, y_true_1, y_true_2, y_true_3 = load_train_model_inputs()
    y_pred_1, y_pred_2, y_pred_3 = infer_model.predict(x_batch)
    
    y_preds = [y_pred_1, y_pred_2, y_pred_3]
    ys = [y_true_1, y_true_2, y_true_3]
    scales = [1, 2, 3]
    
    for i in range(3):
        loss_model = create_loss_model(scales[i])
        loss_value = loss_model.predict([x_batch, t_batch, ys[i], y_preds[i]])
        print("scale: {}, loss_value: {}".format(i+1, loss_value))
        # scale: 1, loss_value: [0.56469357 5.286211  ]
        # scale: 2, loss_value: [0.05866125 4.778614  ]
        # scale: 3, loss_value: [ 0.54328686 11.0839405 ]
    
    loss_value = loss_fn(y_pred_1, y_true_1, scale=1)
    # assert loss_value == [0.56469357, 5.286211]

#     # should (loss_values == losses)
#     losses = train_model.predict([x_batch, t_batch, y_true_1, y_true_2, y_true_3])
