# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
from yolo.train import train_fn
from yolo.config import ConfigParser

argparser = argparse.ArgumentParser(
    description='train yolo-v3 network')

argparser.add_argument(
    '-c',
    '--config',
    default="configs/svhn.json",
    help='config file')


if __name__ == '__main__':
    args = argparser.parse_args()
    config_parser = ConfigParser(args.config)
    
    # 1. create generator
    train_generator, valid_generator = config_parser.create_generator()
    
    # 2. create model
    model = config_parser.create_model()
 
    # 3. training
    learning_rate, save_dname, n_epoches = config_parser.get_train_params()
    train_fn(model,
             train_generator,
             valid_generator,
             learning_rate=learning_rate,
             save_dname=save_dname,
             num_epoches=n_epoches)

