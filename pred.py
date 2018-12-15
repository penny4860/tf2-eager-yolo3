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
    from yolo.utils.visualization_utils import visualize_boxes_and_labels_on_image_array

    category_index = {}
    print(config_parser.get_labels())
    for id_, label_name in enumerate(config_parser.get_labels()):
        category_index[id_] = {"name": label_name}
    print(category_index)
    
    import numpy as np
    boxes = np.array([np.array([b[1],b[0],b[3],b[2]]) for b in boxes])
    print(boxes.shape)
    visualize_boxes_and_labels_on_image_array(image, boxes, labels, probs, category_index)

    # 5. plot    
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()

 


