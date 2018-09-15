# -*- coding: utf-8 -*-

import numpy as np
import struct

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

    def _load_1d_var(self, variable):
        size = np.prod(variable.shape)
        value  = self._read_bytes(size) # bias
        variable.assign(value)

    def load_weights(self, model):

        for i in range(model.num_layers):
            variables = model.get_variables(layer_idx=i, suffix="beta")
            if variables:
                self._load_1d_var(variables[0])

            variables = model.get_variables(layer_idx=i, suffix="gamma")
            if variables:
                self._load_1d_var(variables[0])

            variables = model.get_variables(layer_idx=i, suffix="moving_mean")
            if variables:
                self._load_1d_var(variables[0])

            variables = model.get_variables(layer_idx=i, suffix="moving_variance")
            if variables:
                self._load_1d_var(variables[0])

            variables = model.get_variables(layer_idx=i, suffix="bias")
            if variables:
                self._load_1d_var(variables[0])

            variables = model.get_variables(layer_idx=i, suffix="kernel")
            if variables:
                kernel = variables[0]
                # convolution layer kernel
                if len(kernel.shape) == 4:
                    size = np.prod(kernel.shape)
                    value  = self._read_bytes(size) # scale
                    value = value.reshape(list(reversed(kernel.shape)))
                    value = value.transpose([2,3,1,0])
                    kernel.assign(value)
                # fc layer kernel
                else:
                    size = np.prod(kernel.shape)
                    value  = self._read_bytes(size) # scale
                    value = value.reshape(list(reversed(kernel.shape)))
                    value = value.transpose([1,0])
                    kernel.assign(value)
    
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


if __name__ == '__main__':
    import tensorflow as tf
    tf.enable_eager_execution()
    import os
    from yolo.darknet53 import Darknet53
    from yolo import PROJECT_ROOT
    WEIGHT_FILE = os.path.join(os.path.dirname(PROJECT_ROOT), "dataset", "yolo", "darknet53.weights")

    darknet = Darknet53()
    reader = WeightReader(WEIGHT_FILE)
    reader.load_weights(darknet)




