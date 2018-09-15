# -*- coding: utf-8 -*-

import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models


class Darknet53(tf.keras.Model):
    def __init__(self):
        super(Darknet53, self).__init__(name='')
        
        # (256, 256, 3)
        self.l0a = _ConvBlock(32, stage="0", block="a")
        self.l0_pool = _ConvPoolBlock(64, stage="0", block="b")

        # (128, 128, 64)
        self.l1a = _ResidualBlock([32, 64], stage="1", block="a")
        self.l1_pool = _ConvPoolBlock(128, stage="1", block="b")

        # (64, 64, 128)
        self.l2a = _ResidualBlock([64, 128], stage="2", block="a")
        self.l2b = _ResidualBlock([64, 128], stage="2", block="b")
        self.l2_pool = _ConvPoolBlock(256, stage="2", block="c")

        # (32, 32, 256)
        self.l3a = _ResidualBlock([128, 256], stage="3", block="a")
        self.l3b = _ResidualBlock([128, 256], stage="3", block="b")
        self.l3c = _ResidualBlock([128, 256], stage="3", block="c")
        self.l3d = _ResidualBlock([128, 256], stage="3", block="d")
        self.l3e = _ResidualBlock([128, 256], stage="3", block="e")
        self.l3f = _ResidualBlock([128, 256], stage="3", block="f")
        self.l3g = _ResidualBlock([128, 256], stage="3", block="g")
        self.l3h = _ResidualBlock([128, 256], stage="3", block="h")
        self.l3_pool = _ConvPoolBlock(512, stage="3", block="i")
        
        # (16, 16, 512)
        self.l4a = _ResidualBlock([256, 512], stage="4", block="a")
        self.l4b = _ResidualBlock([256, 512], stage="4", block="b")
        self.l4c = _ResidualBlock([256, 512], stage="4", block="c")
        self.l4d = _ResidualBlock([256, 512], stage="4", block="d")
        self.l4e = _ResidualBlock([256, 512], stage="4", block="e")
        self.l4f = _ResidualBlock([256, 512], stage="4", block="f")
        self.l4g = _ResidualBlock([256, 512], stage="4", block="g")
        self.l4h = _ResidualBlock([256, 512], stage="4", block="h")
        self.l4_pool = _ConvPoolBlock(1024, stage="4", block="i")

        # (8, 8, 1024)
        self.l5a = _ResidualBlock([512, 1024], stage="5", block="a")
        self.l5b = _ResidualBlock([512, 1024], stage="5", block="b")
        self.l5c = _ResidualBlock([512, 1024], stage="5", block="c")
        self.l5d = _ResidualBlock([512, 1024], stage="5", block="d")
        
        # (8, 8, 1024) => (1, 1, 1024)
        self.avg_pool = layers.GlobalAveragePooling2D()
        # self.avg_pool = layers.AveragePooling2D((8, 8), strides=(8, 8))

        # (1, 1, 1024) => (1024)
        self.flatten = layers.Flatten()
        # (1024) => (1000)
        self.fc = layers.Dense(1000, activation='softmax', name='fc1000')

    def call(self, input_tensor, training=False):
        
        x = self.l0a(input_tensor, training)
        x = self.l0_pool(x, training)

        x = self.l1a(x, training)
        x = self.l1_pool(x, training)

        x = self.l2a(x, training)
        x = self.l2b(x, training)
        x = self.l2_pool(x, training)

        x = self.l3a(x, training)
        x = self.l3b(x, training)
        x = self.l3c(x, training)
        x = self.l3d(x, training)
        x = self.l3e(x, training)
        x = self.l3f(x, training)
        x = self.l3g(x, training)
        x = self.l3h(x, training)
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
        
        x = self.flatten(x)
        x = self.fc(x)
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


if __name__ == '__main__':
#     # https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/resnet50/resnet50.py
#     model = make_yolov3_model(256, 256)
#     model.summary()
    tf.enable_eager_execution()

    import numpy as np
    imgs = np.random.randn(1, 256, 256, 3).astype(np.float32)
    input_tensor = tf.constant(imgs)
    darknet = Darknet53()
    y = darknet(input_tensor)
    print(y.shape)
    print(y.numpy().sum())
    
    darknet.variables[0].assign(np.ones((3, 3, 3, 32)))
    for v in darknet.variables[:15]:
        np_kernel = v.numpy()
        print(v.name, np_kernel.shape, np_kernel[0,0,0,:5])
        break
        

