# -*- coding: utf-8 -*-

import functools
import tensorflow as tf

layers = tf.keras.layers


class _IdentityBlock(tf.keras.Model):
    """_IdentityBlock is the block that has no conv layer at shortcut.

    Args:
        kernel_size: the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        data_format: data_format for the input ('channels_first' or
            'channels_last').
    """

    def __init__(self, kernel_size, filters, stage, block, data_format="channels_last"):
        super(_IdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else 3

        self.conv2a = layers.Conv2D(
                filters1, (1, 1), name=conv_name_base + '2a', data_format=data_format)
        self.bn2a = layers.BatchNormalization(
                axis=bn_axis, name=bn_name_base + '2a')

        self.conv2b = layers.Conv2D(
                filters2,
                kernel_size,
                padding='same',
                data_format=data_format,
                name=conv_name_base + '2b')
        self.bn2b = layers.BatchNormalization(
                axis=bn_axis, name=bn_name_base + '2b')

        self.conv2c = layers.Conv2D(
                filters3, (1, 1), name=conv_name_base + '2c', data_format=data_format)
        self.bn2c = layers.BatchNormalization(
                axis=bn_axis, name=bn_name_base + '2c')

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


class _ConvBlock(tf.keras.Model):
    """_ConvBlock is the block that has a conv layer at shortcut.

    Args:
            kernel_size: the kernel size of middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            data_format: data_format for the input ('channels_first' or
                'channels_last').
            strides: strides for the convolution. Note that from stage 3, the first
             conv layer at main path is with strides=(2,2), and the shortcut should
             have strides=(2,2) as well.
    """

    def __init__(self,
                             kernel_size,
                             filters,
                             stage,
                             block,
                             data_format="channels_last",
                             strides=(2, 2)):
        super(_ConvBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'
        bn_axis = 1 if data_format == 'channels_first' else 3

        self.conv2a = layers.Conv2D(
                filters1, (1, 1),
                strides=strides,
                name=conv_name_base + '2a',
                data_format=data_format)
        self.bn2a = layers.BatchNormalization(
                axis=bn_axis, name=bn_name_base + '2a')

        self.conv2b = layers.Conv2D(
                filters2,
                kernel_size,
                padding='same',
                name=conv_name_base + '2b',
                data_format=data_format)
        self.bn2b = layers.BatchNormalization(
                axis=bn_axis, name=bn_name_base + '2b')

        self.conv2c = layers.Conv2D(
                filters3, (1, 1), name=conv_name_base + '2c', data_format=data_format)
        self.bn2c = layers.BatchNormalization(
                axis=bn_axis, name=bn_name_base + '2c')

        self.conv_shortcut = layers.Conv2D(
                filters3, (1, 1),
                strides=strides,
                name=conv_name_base + '1',
                data_format=data_format)
        self.bn_shortcut = layers.BatchNormalization(
                axis=bn_axis, name=bn_name_base + '1')

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)
        x = self.bn2c(x, training=training)

        shortcut = self.conv_shortcut(input_tensor)
        shortcut = self.bn_shortcut(shortcut, training=training)

        x += shortcut
        return tf.nn.relu(x)


# pylint: disable=not-callable
class ResNet50(tf.keras.Model):
    """Instantiates the ResNet50 architecture.

    Args:
        data_format: format for the image. Either 'channels_first' or
            'channels_last'.    'channels_first' is typically faster on GPUs while
            'channels_last' is typically faster on CPUs. See
            https://www.tensorflow.org/performance/performance_guide#data_formats
        include_top: whether to include the fully-connected layer at the top of the
            network.
        classes: optional number of classes to classify images into, only to be
            specified if `include_top` is True.

    Raises:
            ValueError: in case of invalid argument for data_format.
    """

    def __init__(self,
                 include_top=True,
                 classes=1000):
        super(ResNet50, self).__init__(name='')
        self.include_top = include_top
        
        # Stage 1 : (224, 224, 3) => (56, 56, 64)
        self.conv1 = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')
        self.bn_conv1 = layers.BatchNormalization(axis=3, name='bn_conv1')
        self.max_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))

        # Stage 2 : (56, 56, 64) => (56, 56, 256)
        self.l2a = _ConvBlock(3, [64, 64, 256], stage=2, block="a", strides=(1,1))
        self.l2b = _IdentityBlock(3, [64, 64, 256], stage=2, block="b")
        self.l2c = _IdentityBlock(3, [64, 64, 256], stage=2, block="c")
        
        # Stage 3 : (56, 56, 256) => (28, 28, 256) => (28, 28, 512)
        # max-pooling 을 생략하는 대신 first block 에서 strides size 를 2로 한다. -> spatial size를 reduce
        self.l3a = _ConvBlock(3, [128, 128, 512], stage=3, block="a", strides=(2,2))
        self.l3b = _IdentityBlock(3, [128, 128, 512], stage=3, block="b")
        self.l3c = _IdentityBlock(3, [128, 128, 512], stage=3, block="c")
        self.l3d = _IdentityBlock(3, [128, 128, 512], stage=3, block="d")

        # Stage 4 : (28, 28, 512) => (14, 14, 512) => (14, 14, 1024)
        self.l4a = _ConvBlock(3, [256, 256, 1024], stage=4, block="a", strides=(2,2))
        self.l4b = _IdentityBlock(3, [256, 256, 1024], stage=4, block="b")
        self.l4c = _IdentityBlock(3, [256, 256, 1024], stage=4, block="c")
        self.l4d = _IdentityBlock(3, [256, 256, 1024], stage=4, block="d")
        self.l4e = _IdentityBlock(3, [256, 256, 1024], stage=4, block="e")
        self.l4f = _IdentityBlock(3, [256, 256, 1024], stage=4, block="f")

        # Stage 5 : (14, 14, 1024) => (7, 7, 1024) => (7, 7, 2048)
        self.l5a = _ConvBlock(3, [512, 512, 2048], stage=5, block='a', strides=(2,2))
        self.l5b = _IdentityBlock(3, [512, 512, 2048], stage=5, block='b')
        self.l5c = _IdentityBlock(3, [512, 512, 2048], stage=5, block='c')

        # (7, 7, 2048) => (1, 1, 2048)
        self.avg_pool = layers.AveragePooling2D((7, 7), strides=(7, 7))

        if self.include_top:
            # (1, 1, 2048) => (2048)
            self.flatten = layers.Flatten()
            # (2048) => (1000)
            self.fc1000 = layers.Dense(classes, name='fc1000')
        else:
            reduction_indices = [1, 2]
            reduction_indices = tf.constant(reduction_indices)
            self.global_pooling = functools.partial(
                    tf.reduce_mean,
                    reduction_indices=reduction_indices,
                    keep_dims=False)

    def call(self, input_tensor, training):
        x = self.conv1(input_tensor)
        x = self.bn_conv1(x, training=training)
        x = tf.nn.relu(x)
        x = self.max_pool(x)

        x = self.l2a(x, training=training)
        x = self.l2b(x, training=training)
        x = self.l2c(x, training=training)

        x = self.l3a(x, training=training)
        x = self.l3b(x, training=training)
        x = self.l3c(x, training=training)
        x = self.l3d(x, training=training)

        x = self.l4a(x, training=training)
        x = self.l4b(x, training=training)
        x = self.l4c(x, training=training)
        x = self.l4d(x, training=training)
        x = self.l4e(x, training=training)
        x = self.l4f(x, training=training)

        x = self.l5a(x, training=training)
        x = self.l5b(x, training=training)
        x = self.l5c(x, training=training)

        x = self.avg_pool(x)

        if self.include_top:
            return self.fc1000(self.flatten(x))
        elif self.global_pooling:
            return self.global_pooling(x)
        else:
            return x


if __name__ == "__main__":
    model = ResNet50()
    print(model)
    
    import numpy as np
    imgs = np.random.randn(1, 224, 224, 3).astype(np.float32)
    input_tensor = tf.constant(imgs)
    print(input_tensor)
    
    output_tensor = model(input_tensor, training=False)
    print(output_tensor.shape)

