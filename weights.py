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


if __name__ == '__main__':
    import tensorflow as tf
    tf.enable_eager_execution()
    import os
    from yolo.darknet53 import Darknet53
    from yolo import PROJECT_ROOT
    WEIGHT_FILE = os.path.join(os.path.dirname(PROJECT_ROOT), "dataset", "yolo", "darknet53.weights")


    darknet = Darknet53()


