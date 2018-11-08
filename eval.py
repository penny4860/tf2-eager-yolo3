# -*- coding: utf-8 -*-
# Todo : eval.py 에서 config parser를 사용
import tensorflow as tf
tf.enable_eager_execution()

import argparse

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
    from yolo.config import ConfigParser
    from yolo.frontend import Evaluator
    args = argparser.parse_args()
    config_parser = ConfigParser(args.config)
    model = config_parser.create_model()
    detector = config_parser.create_detector(model)
 
    evaluator = Evaluator(detector, config_parser.get_labels())
    score = evaluator.run(config_parser.get_train_anns(),
                          config_parser._train_config["train_image_folder"])
    
    print(score)

