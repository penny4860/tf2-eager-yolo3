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
    '-i',
    '--image',
    default="tests/samples/sample.jpeg",
    help='path to image file')


if __name__ == '__main__':
    args = argparser.parse_args()

    image_path   = args.image
    # Download if not exits weight file
    weights_path = "yolov3.weights"
    download_if_not_exists(weights_path,
                           "https://pjreddie.com/media/files/yolov3.weights")

    # set some parameters
    labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
              "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
              "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
              "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

    # 1. create yolo model & load weights
    yolov3 = Yolonet()
    yolov3.load_darknet_params(weights_path)

    # 2. preprocess the image
    image = cv2.imread(image_path)
    image = image[:,:,::-1]

    d = YoloDetector(yolov3)
    boxes = d.detect(image, COCO_ANCHORS, net_size=416)
    
    # 4. draw detected boxes
    image = draw_boxes(image, boxes, labels, obj_thresh=0.5)

    # 5. plot    
    plt.imshow(image)
    plt.show()


 


