# -*- coding: utf-8 -*-

import tensorflow as tf
import os
import pytest
import glob

from yolo import PROJECT_ROOT
from yolo.utils.utils import download_if_not_exists

@pytest.fixture(scope='session')
def setup_tf_eager(request):
    tf.enable_eager_execution()


@pytest.fixture(scope='session')
def setup_darknet_weights():
    weights_path = os.path.join(PROJECT_ROOT, "tests", "samples", "yolov3.weights")
    download_if_not_exists(weights_path,
                           "https://pjreddie.com/media/files/yolov3.weights")

    return weights_path


@pytest.fixture(scope='session')
def setup_train_dirs():
    img_root = os.path.join(PROJECT_ROOT, "tests", "dataset", "raccoon", "imgs")
    ann_dir = os.path.join(PROJECT_ROOT, "tests", "dataset", "raccoon", "anns")
    ann_fnames = glob.glob(os.path.join(ann_dir, "*.xml"))
    return ann_fnames, img_root


