# -*- coding: utf-8 -*-

import tensorflow as tf
tf.enable_eager_execution()
import matplotlib.pyplot as plt
import argparse
import cv2

from yolo.utils.utils import download_if_not_exists
from yolo.utils.box import draw_boxes
from yolo.net.yolonet import Yolonet
from yolo.frontend import YoloDetector


argparser = argparse.ArgumentParser(
    description='test yolov3 network with coco weights')

argparser.add_argument(
    '-c',
    '--config',
    default="configs/predict_coco.json",
    help='config file')

argparser.add_argument(
    '-i',
    '--image',
    default="tests/samples/sample.jpeg",
    help='path to image file')


if __name__ == '__main__':
    args = argparser.parse_args()
    image_path   = args.image
    
    import json
    with open(args.config) as data_file:    
        config = json.load(data_file)
    
    # Download if not exits weight file
    # 1. create yolo model & load weights
    from yolo.config import ConfigParser
    config_parser = ConfigParser(args.config)
    model = config_parser.create_model()
    print(model)
    
    # 2. preprocess the image
    image = cv2.imread(image_path)
    image = image[:,:,::-1]
    print(image.shape)

    d = YoloDetector(model, config["model"]["anchors"], net_size=config["model"]["net_size"])
    boxes, labels, probs = d.detect(image, 0.5)
    print(boxes)
    
    # 4. draw detected boxes
    image = draw_boxes(image, boxes, labels, probs, config["model"]["labels"])

    # 5. plot    
    plt.imshow(image)
    plt.show()


 


