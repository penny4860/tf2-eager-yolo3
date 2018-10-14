# -*- coding: utf-8 -*-

from yolo.post_proc.decoder import postprocess_ouput
from yolo.net.yolonet import preprocess_input


class YoloDetector(object):
    
    def __init__(self, model):
        self._model = model
        
    def detect(self, image, anchors, net_size=288):
        image_h, image_w, _ = image.shape
        new_image = preprocess_input(image, net_size)
        # 3. predict
        yolos = self._model.predict(new_image)
        boxes = postprocess_ouput(yolos, anchors, net_size, image_h, image_w)
        return boxes



