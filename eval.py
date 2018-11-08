# -*- coding: utf-8 -*-
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
    args = argparser.parse_args()
    config_parser = ConfigParser(args.config)
    model = config_parser.create_model()
    evaluator, _ = config_parser.create_evaluator(model)

    score = evaluator.run(threshold=args.threshold,
                          save_dname=args.save_dname)
    
    print(score)

