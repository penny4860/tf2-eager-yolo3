# -*- coding: utf-8 -*-

import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models


# Darknet53 feature extractor
class Headnet(tf.keras.Model):
    def __init__(self):
        super(Headnet, self).__init__(name='')
#         # self.l5d = _ResidualBlock([512, 1024], layer_idx=[50, 51], name="stage5")
#         self.num_layers = 52
#         self._init_vars()

    def call(self, input_tensor, training=False):
#         x = self.l0a(input_tensor, training)
        pass

    def get_variables(self, layer_idx, suffix=None):
        if suffix:
            find_name = "layer_{}/{}".format(layer_idx, suffix)
        else:
            find_name = "layer_{}/".format(layer_idx)
        variables = []
        for v in self.variables:
            if find_name in v.name:
                variables.append(v)
        return variables

    def _init_vars(self):
        import numpy as np
        imgs = np.random.randn(1, 256, 256, 3).astype(np.float32)
        input_tensor = tf.constant(imgs)
        self.call(input_tensor)


class _ConvBlock(tf.keras.Model):
    def __init__(self, filters, layer_idx, name=""):
        super(_ConvBlock, self).__init__(name=name)
        
        layer_name = "layer_{}".format(str(layer_idx))

        self.conv = layers.Conv2D(filters, (3, 3), strides=(1, 1), padding='same', use_bias=False, name=layer_name)
        self.bn = layers.BatchNormalization(epsilon=0.001, name=layer_name)

    def call(self, input_tensor, training=False):

        x = self.conv(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        return x


if __name__ == '__main__':
    import numpy as np
    tf.enable_eager_execution()
    imgs = np.random.randn(1, 256, 256, 3).astype(np.float32)
    input_tensor = tf.constant(imgs)
    
    # (1, 256, 256, 3) => (1, 8, 8, 1024)
    bodynet = Bodynet()
    s3, s4, s5 = bodynet(input_tensor)
    print(s3.shape, s4.shape, s5.shape)

