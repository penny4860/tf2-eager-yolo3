# -*- coding: utf-8 -*-

import tensorflow as tf
tf.enable_eager_execution()
import argparse
import cv2

from yolo.utils.box import draw_boxes
from yolo.config import ConfigParser


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
    
    # 1. create yolo model & load weights
    config_parser = ConfigParser(args.config)
    model = config_parser.create_model(skip_detect_layer=False)
    detector = config_parser.create_detector(model)
    
    # 2. Load image
    image = cv2.imread(image_path)
    image = image[:,:,::-1]
    
    # 3. Run detection
    boxes, labels, probs = detector.detect(image, 0.5)
    
    # 4. draw detected boxes
    image = draw_boxes(image, boxes, labels, probs, config_parser.get_labels())

    # 5. plot    
    cv2.imwrite("detected.jpg", image[:,:,::-1])

 


