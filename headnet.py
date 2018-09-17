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


class _Conv2(tf.keras.Model):
    def __init__(self, filters, layer_idx, name=""):
        super(_Conv2, self).__init__(name=name)
        
        layer_names = ["layer_{}".format(i) for i in layer_idx]

        self.conv1 = layers.Conv2D(filters[0], (3, 3), strides=(1, 1), padding='same', use_bias=False, name=layer_names[0])
        self.bn = layers.BatchNormalization(epsilon=0.001, name=layer_names[0])
        self.conv2 = layers.Conv2D(filters[1], (1, 1), strides=(1, 1), padding='same', use_bias=True, name=layer_names[1])

    def call(self, input_tensor, training=False):
        x = self.conv1(input_tensor)
        x = self.bn(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.conv2(input_tensor, training)
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

