# -*- coding: utf-8 -*-

import tensorflow as tf
tf.enable_eager_execution()

import matplotlib.pyplot as plt
import cv2
import os
import argparse
import json
import glob

from yolo.net import Yolonet
from yolo.train import train
from yolo.frontend import YoloDetector
from yolo.utils.box import draw_boxes
from yolo.dataset.generator import create_generator


argparser = argparse.ArgumentParser(
    description='train yolo-v3 network')

argparser.add_argument(
    '-c',
    '--config',
    default="configs/raccoon.json",
    help='config file')


if __name__ == '__main__':
    args = argparser.parse_args()
    with open(args.config) as data_file:    
        config = json.load(data_file)
    
    # 1. create generator
    ann_fnames = glob.glob(os.path.join(config["train"]["train_annot_folder"], "*.xml"))
    generator = create_generator(ann_fnames,
                                 config["train"]["train_image_folder"],
                                 batch_size=config["train"]["batch_size"],
                                 labels_naming=config["model"]["labels"],
                                 anchors=config["model"]["anchors"],
                                 jitter=config["train"]["jitter"])
    # 2. create model
    model = Yolonet(n_classes=len(config["model"]["labels"]))
    model.load_darknet_params(config["pretrained"]["darknet_format"], skip_detect_layer=True)
 
    # 4. training
    train(generator,
          model,
          learning_rate=config["train"]["learning_rate"],
          save_dname=config["train"]["save_folder"],
          num_epoches=config["train"]["num_epoch"],
          verbose=1)

    # 5. prepare sample images
    img_fnames = glob.glob(os.path.join(config["train"]["train_image_folder"], "*.*"))
    imgs = [cv2.imread(fname)[:,:,::-1] for fname in img_fnames]

    # 6. create new model & load trained weights
    model = Yolonet(n_classes=len(config["model"]["labels"]))
    model.load_weights(os.path.join(config["train"]["save_folder"], "weights.h5"))
    detector = YoloDetector(model)
 
    # 7. predict & plot
    boxes = detector.detect(imgs[0], config["model"]["anchors"])
    image = draw_boxes(imgs[0], boxes, labels=config["model"]["labels"])
    plt.imshow(image)
    plt.show()


