# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import struct

layers = tf.keras.layers
models = tf.keras.models


class WeightReader:
    def __init__(self, weight_file):
        with open(weight_file, 'rb') as w_f:
            major,    = struct.unpack('i', w_f.read(4))
            minor,    = struct.unpack('i', w_f.read(4))
            _, = struct.unpack('i', w_f.read(4))

            if (major*10 + minor) >= 2 and major < 1000 and minor < 1000:
                w_f.read(8)
            else:
                w_f.read(4)
            binary = w_f.read()

        self.offset = 0
        self.all_weights = np.frombuffer(binary, dtype='float32')

    def load_weights(self, model):
        for i in range(106):
            try:
                conv_layer = model.get_layer('conv_' + str(i))
                print("loading weights of convolution #" + str(i))

                # batch norm layer 가 있는 경우
                if i not in [81, 93, 105]:
                    self._load_bn_params(model, i)

                # conv layer : bias 가 있는 경우
                if len(conv_layer.get_weights()) > 1:
                    self._load_conv_bias_params(conv_layer)
                # conv layer : bias 가 없는 경우
                else:
                    self._load_conv_params(conv_layer)

            except ValueError:
                print("no convolution #" + str(i))     
    
    def _read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
    
    def _load_bn_params(self, model, i):
        norm_layer = model.get_layer('bnorm_' + str(i))

        size = np.prod(norm_layer.get_weights()[0].shape)
        print(len(norm_layer.get_weights()), norm_layer.get_weights()[0].shape, size)

        beta  = self._read_bytes(size) # bias
        gamma = self._read_bytes(size) # scale
        mean  = self._read_bytes(size) # mean
        var   = self._read_bytes(size) # variance            

        norm_layer.set_weights([gamma, beta, mean, var])  

    def _load_conv_bias_params(self, conv_layer):
        bias   = self._read_bytes(np.prod(conv_layer.get_weights()[1].shape))
        kernel = self._read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2,3,1,0])
        conv_layer.set_weights([kernel, bias])

    def _load_conv_params(self, conv_layer):
        kernel = self._read_bytes(np.prod(conv_layer.get_weights()[0].shape))
        kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
        kernel = kernel.transpose([2,3,1,0])
        conv_layer.set_weights([kernel])


def _conv_block(inp, convs, skip=True):
    x = inp
    count = 0
    
    for conv in convs:
        if count == (len(convs) - 2) and skip:
            skip_connection = x
        count += 1
        
        if conv['stride'] > 1: x = layers.ZeroPadding2D(((1,0),(1,0)))(x) # peculiar padding as darknet prefer left and top
        x = layers.Conv2D(conv['filter'], 
                          conv['kernel'], 
                          strides=conv['stride'], 
                          padding='valid' if conv['stride'] > 1 else 'same', # peculiar padding as darknet prefer left and top
                          name='conv_' + str(conv['layer_idx']), 
                          use_bias=False if conv['bnorm'] else True)(x)
        if conv['bnorm']: x = layers.BatchNormalization(epsilon=0.001, name='bnorm_' + str(conv['layer_idx']))(x)
        if conv['leaky']: x = layers.LeakyReLU(alpha=0.1, name='leaky_' + str(conv['layer_idx']))(x)

    return layers.add([skip_connection, x]) if skip else x


def make_yolov3_model(h = None, w = None):
    input_image = layers.Input(shape=(h, w, 3))

    # Layer  0 => 4
    x = _conv_block(input_image, [{'filter': 32, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 0},
                                  {'filter': 64, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 1},
                                  {'filter': 32, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 2},
                                  {'filter': 64, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 3}])

    # Layer  5 => 8
    x = _conv_block(x, [{'filter': 128, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 5},
                        {'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 6},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 7}])

    # Layer  9 => 11
    x = _conv_block(x, [{'filter':  64, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 9},
                        {'filter': 128, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 10}])

    # Layer 12 => 15
    x = _conv_block(x, [{'filter': 256, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 12},
                        {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 13},
                        {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 14}])

    # Layer 16 => 36
    for i in range(7):
        x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 16+i*3},
                            {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 17+i*3}])
        
    skip_36 = x
        
    # Layer 37 => 40
    x = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 37},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 38},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 39}])

    # Layer 41 => 61
    for i in range(7):
        x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 41+i*3},
                            {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 42+i*3}])
        
    skip_61 = x
        
    # Layer 62 => 65
    x = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 2, 'bnorm': True, 'leaky': True, 'layer_idx': 62},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 63},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 64}])

    # Layer 66 => 74
    for i in range(3):
        x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 66+i*3},
                            {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 67+i*3}])
    # 74 layer : final convolutional layer
    # Layer 75 => 79
    x = _conv_block(x, [{'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 75},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 76},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 77},
                        {'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 78},
                        {'filter':  512, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 79}], skip=False)

    # Layer 80 => 82
    yolo_82 = _conv_block(x, [{'filter': 1024, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 80},
                              {'filter':  255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 81}], skip=False)

    # Layer 83 => 86
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 84}], skip=False)
    x = layers.UpSampling2D(2)(x)
    x = layers.concatenate([x, skip_61])

    # Layer 87 => 91
    x = _conv_block(x, [{'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 87},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 88},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 89},
                        {'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 90},
                        {'filter': 256, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True, 'layer_idx': 91}], skip=False)

    # Layer 92 => 94
    yolo_94 = _conv_block(x, [{'filter': 512, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 92},
                              {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 93}], skip=False)

    # Layer 95 => 98
    x = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True, 'leaky': True,   'layer_idx': 96}], skip=False)
    x = layers.UpSampling2D(2)(x)
    x = layers.concatenate([x, skip_36])

    # Layer 99 => 106
    yolo_106 = _conv_block(x, [{'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 99},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 100},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 101},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 102},
                               {'filter': 128, 'kernel': 1, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 103},
                               {'filter': 256, 'kernel': 3, 'stride': 1, 'bnorm': True,  'leaky': True,  'layer_idx': 104},
                               {'filter': 255, 'kernel': 1, 'stride': 1, 'bnorm': False, 'leaky': False, 'layer_idx': 105}], skip=False)

    model = models.Model(input_image, [yolo_82, yolo_94, yolo_106])    
    return model


class _ConvBlock5(tf.keras.Model):
    def __init__(self, filters, stage, block):
        super(_ConvBlock5, self).__init__(name='')
        filters1, filters2, filters3, filters4, filters5 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        self.conv2a = layers.Conv2D(filters1, (1, 1), padding='same', use_bias=False, name=conv_name_base + '2a')
        self.bn2a = layers.BatchNormalization(epsilon=0.001, name=bn_name_base + '2a')

        self.conv2b = layers.Conv2D(filters2, (3, 3), padding='same', use_bias=False, name=conv_name_base + '2b')
        self.bn2b = layers.BatchNormalization(epsilon=0.001, name=bn_name_base + '2b')

        self.conv2c = layers.Conv2D(filters3, (1, 1), padding='same', use_bias=False, name=conv_name_base + '2c')
        self.bn2c = layers.BatchNormalization(epsilon=0.001, name=bn_name_base + '2c')

        self.conv2d = layers.Conv2D(filters4, (3, 3), padding='same', use_bias=False, name=conv_name_base + '2d')
        self.bn2d = layers.BatchNormalization(epsilon=0.001, name=bn_name_base + '2d')

        self.conv2e = layers.Conv2D(filters5, (1, 1), padding='same', use_bias=False, name=conv_name_base + '2e')
        self.bn2e = layers.BatchNormalization(epsilon=0.001, name=bn_name_base + '2e')


    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        x = self.conv2c(x)
        x = self.bn2c(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        x = self.conv2d(x)
        x = self.bn2d(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        x = self.conv2e(x)
        x = self.bn2e(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        return x


class _ConvBlock2(tf.keras.Model):
    def __init__(self, filters, stage, block):
        super(_ConvBlock2, self).__init__(name='')
        filters1, filters2 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        self.conv2a = layers.Conv2D(filters1, (1, 1), padding='same', use_bias=False, name=conv_name_base + '2a')
        self.bn2a = layers.BatchNormalization(epsilon=0.001, name=bn_name_base + '2a')

        self.conv2b = layers.Conv2D(filters2, (3, 3), padding='same', use_bias=False, name=conv_name_base + '2b')
        self.bn2b = layers.BatchNormalization(epsilon=0.001, name=bn_name_base + '2b')

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        
        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        return x

def get_darknet():
    from yolo import YOLOV3_WEIGHTS
    
    yolov3 = make_yolov3_model(256, 256)
    weight_reader = WeightReader(YOLOV3_WEIGHTS)
    weight_reader.load_weights(yolov3)

    # model.summary()
    input_tensor = yolov3.get_layer("input_1")
    output_tensor = yolov3.get_layer("leaky_73")
    
    darknet = models.Model(input_tensor.input,
                           output_tensor.output)
    return darknet

if __name__ == '__main__':
    # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/resnet50/resnet50.py
    model = make_yolov3_model(256, 256)
    model.summary()
    

