# -*- coding: utf-8 -*-

import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models


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


class _ConvBlock(tf.keras.Model):
    def __init__(self, filters, stage, block):
        super(_ConvBlock, self).__init__(name='')

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        self.conv = layers.Conv2D(filters, (3, 3), strides=(1, 1), padding='same', use_bias=False, name=conv_name_base)
        self.bn = layers.BatchNormalization(epsilon=0.001, name=bn_name_base)

    def call(self, input_tensor, training=False):

        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        return x


class _ConvPoolBlock(tf.keras.Model):
    def __init__(self, filters, stage, block):
        super(_ConvPoolBlock, self).__init__(name='')

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        self.pad = layers.ZeroPadding2D(((1,0),(1,0)))
        self.conv = layers.Conv2D(filters, (3, 3), strides=(2, 2), padding='valid', use_bias=False, name=conv_name_base)
        self.bn = layers.BatchNormalization(epsilon=0.001, name=bn_name_base)

    def call(self, input_tensor, training=False):

        x = self.pad(input_tensor)
        x = self.conv(x)
        x = self.bn(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        return x


class _ResidualBlock(tf.keras.Model):
    def __init__(self, filters, stage, block):
        super(_ResidualBlock, self).__init__(name='')
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

        x += input_tensor
        return x


class Yolo3(tf.keras.Model):
    def __init__(self):
        super(Yolo3, self).__init__(name='')
        
        # (256, 256, 3)
        self.l1a = _ConvBlock(32, stage="1", block="a")
        self.l1_pool = _ConvPoolBlock(64, stage="1", block="b")

        # (128, 128, 64)
        self.l2a = _ResidualBlock([32, 64], stage="2", block="a")
        self.l2_pool = _ConvPoolBlock(128, stage="2", block="b")

        # (64, 64, 128)
        self.l3a = _ResidualBlock([64, 128], stage="3", block="a")
        self.l3b = _ResidualBlock([64, 128], stage="3", block="b")
        self.l3_pool = _ConvPoolBlock(256, stage="3", block="c")

        # (32, 32, 256)
        self.l4a = _ResidualBlock([128, 256], stage="4", block="a")
        self.l4b = _ResidualBlock([128, 256], stage="4", block="b")
        self.l4c = _ResidualBlock([128, 256], stage="4", block="c")
        self.l4d = _ResidualBlock([128, 256], stage="4", block="d")
        self.l4e = _ResidualBlock([128, 256], stage="4", block="e")
        self.l4f = _ResidualBlock([128, 256], stage="4", block="f")
        self.l4g = _ResidualBlock([128, 256], stage="4", block="g")
        self.l4h = _ResidualBlock([128, 256], stage="4", block="h")
        self.l4_pool = _ConvPoolBlock(512, stage="4", block="i")
        
        # (16, 16, 512)
        self.l5a = _ResidualBlock([256, 512], stage="5", block="a")
        self.l5b = _ResidualBlock([256, 512], stage="5", block="b")
        self.l5c = _ResidualBlock([256, 512], stage="5", block="c")
        self.l5d = _ResidualBlock([256, 512], stage="5", block="d")
        self.l5e = _ResidualBlock([256, 512], stage="5", block="e")
        self.l5f = _ResidualBlock([256, 512], stage="5", block="f")
        self.l5g = _ResidualBlock([256, 512], stage="5", block="g")
        self.l5h = _ResidualBlock([256, 512], stage="5", block="h")
        self.l5_pool = _ConvPoolBlock(1024, stage="5", block="i")

        # (8, 8, 1024)
        self.l6a = _ResidualBlock([512, 1024], stage="6", block="a")
        self.l6b = _ResidualBlock([512, 1024], stage="6", block="b")
        self.l6c = _ResidualBlock([512, 1024], stage="6", block="c")
        self.l6d = _ResidualBlock([512, 1024], stage="6", block="d")


    def call(self, input_tensor, training=False):
        
        x = self.l1a(input_tensor, training)
        x = self.l1_pool(x, training)

        x = self.l2a(x, training)
        x = self.l2_pool(x, training)

        x = self.l3a(x, training)
        x = self.l3b(x, training)
        x = self.l3_pool(x, training)

        x = self.l4a(x, training)
        x = self.l4b(x, training)
        x = self.l4c(x, training)
        x = self.l4d(x, training)
        x = self.l4e(x, training)
        x = self.l4f(x, training)
        x = self.l4g(x, training)
        x = self.l4h(x, training)
        x = self.l4_pool(x, training)

        x = self.l5a(x, training)
        x = self.l5b(x, training)
        x = self.l5c(x, training)
        x = self.l5d(x, training)
        x = self.l5e(x, training)
        x = self.l5f(x, training)
        x = self.l5g(x, training)
        x = self.l5h(x, training)
        x = self.l5_pool(x, training)

        x = self.l6a(x, training)
        x = self.l6b(x, training)
        x = self.l6c(x, training)
        x = self.l6d(x, training)
        return x

if __name__ == '__main__':
#     # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/resnet50/resnet50.py
#     model = make_yolov3_model(256, 256)
#     model.summary()
    
    import numpy as np
    imgs = np.random.randn(1, 256, 256, 3).astype(np.float32)
    input_tensor = tf.constant(imgs)
    yolo = Yolo3()
    y = yolo(input_tensor)
    print(y)
    print(y.shape)

    imgs = np.random.randn(1, 8, 8, 1024).astype(np.float32)
    input_tensor = tf.constant(imgs)
    _conv5_layers = _ConvBlock5([512, 1024, 512, 1024, 512], "5_1", "a")
    y = _conv5_layers(input_tensor)
    print(y)
    print(y.shape)
    
