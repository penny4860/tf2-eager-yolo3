# -*- coding: utf-8 -*-

import os

PKG_ROOT = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(PKG_ROOT)
DARKNET_WEIGHTS = os.path.join(os.path.dirname(PROJECT_ROOT), "dataset", "yolo", "darknet53.weights")    
YOLOV3_WEIGHTS = os.path.join(os.path.dirname(PROJECT_ROOT), "dataset", "yolo", "yolov3.weights")    

