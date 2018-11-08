# -*- coding: utf-8 -*-
# Todo : eval.py 에서 config parser를 사용
import tensorflow as tf
tf.enable_eager_execution()

import cv2
import os
import argparse
import json
from tqdm import tqdm

from yolo.utils.box import draw_boxes

argparser = argparse.ArgumentParser(
    description='evaluate yolo-v3 network')

argparser.add_argument(
    '-c',
    '--config',
    default="configs/svhn.json",
    help='config file')

argparser.add_argument(
    '-s',
    '--save_dname',
    default=None)

argparser.add_argument(
    '-t',
    '--threshold',
    type=float,
    default=0.5)


if __name__ == '__main__':
    from yolo.eval.fscore import count_true_positives, calc_score
    from yolo.config import ConfigParser
    
    args = argparser.parse_args()
    with open(args.config) as data_file:    
        config = json.load(data_file)

    config_parser = ConfigParser(args.config)
    model = config_parser.create_model()
    detector = config_parser.create_detector(model)
 
    n_true_positives = 0
    n_truth = 0
    n_pred = 0
    ann_fnames = config_parser.get_train_anns()
    for ann_fname in tqdm(ann_fnames):
        image, img_fname, true_boxes, true_labels = config_parser.parse_ann(ann_fname)

        boxes, labels, probs = detector.detect(image, args.threshold)
        
        n_true_positives += count_true_positives(boxes, true_boxes, labels, true_labels)
        n_truth += len(true_boxes)
        n_pred += len(boxes)
        
        if args.save_dname:
            image_ = draw_boxes(image, boxes, labels, probs, config_parser.get_labels(), desired_size=416)
            output_path = os.path.join(args.save_dname, os.path.split(img_fname)[-1])
            cv2.imwrite(output_path, image_[:,:,::-1])

    print(calc_score(n_true_positives, n_truth, n_pred))


