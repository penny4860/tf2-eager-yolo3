# -*- coding: utf-8 -*-

import tensorflow as tf
tf.enable_eager_execution()
import matplotlib.pyplot as plt
import argparse
import cv2

from yolo.utils.utils import download_if_not_exists
from yolo.utils.box import draw_boxes
from yolo.net.yolonet import Yolonet
from yolo import COCO_ANCHORS
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
    download_if_not_exists(config["pretrained"]["darknet_format"],
                           "https://pjreddie.com/media/files/yolov3.weights")

    # 1. create yolo model & load weights
    yolov3 = Yolonet()
    yolov3.load_darknet_params(config["pretrained"]["darknet_format"])

    # 2. preprocess the image
    image = cv2.imread(image_path)
    image = image[:,:,::-1]

    d = YoloDetector(yolov3)
    boxes = d.detect(image, COCO_ANCHORS, net_size=416)
    
    # 4. draw detected boxes
    image = draw_boxes(image, boxes, config["model"]["labels"], obj_thresh=0.5)

    # 5. plot    
    plt.imshow(image)
    plt.show()


 


