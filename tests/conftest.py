# -*- coding: utf-8 -*-

import tensorflow as tf
import pytest

@pytest.fixture(scope='session')
def setup_tf_eager(request):
    tf.enable_eager_execution()

