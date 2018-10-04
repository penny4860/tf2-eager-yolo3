# -*- coding: utf-8 -*-

from yolo import PROJECT_ROOT
import os
import numpy as np
TEST_SAMPLE_ROOT = os.path.join(PROJECT_ROOT, "tests", "samples")

def sample_list_y_trues():
    # (2, 9, 9, 3, 6) (2, 18, 18, 3, 6) (2, 36, 36, 3, 6)
    y_true_1 = np.load(os.path.join(TEST_SAMPLE_ROOT, "yolo_1.npy")).astype(np.float32)
    y_true_2 = np.load(os.path.join(TEST_SAMPLE_ROOT, "yolo_2.npy")).astype(np.float32)
    y_true_3 = np.load(os.path.join(TEST_SAMPLE_ROOT, "yolo_3.npy")).astype(np.float32)
    y_trues = [y_true_1, y_true_2, y_true_3]
    return y_trues

def sample_list_y_preds():
    # (2, 9, 9, 3, 6) (2, 18, 18, 3, 6) (2, 36, 36, 3, 6)
    y_pred_1 = np.load(os.path.join(TEST_SAMPLE_ROOT, "y_pred_1.npy")).astype(np.float32)
    y_pred_2 = np.load(os.path.join(TEST_SAMPLE_ROOT, "y_pred_2.npy")).astype(np.float32)
    y_pred_3 = np.load(os.path.join(TEST_SAMPLE_ROOT, "y_pred_3.npy")).astype(np.float32)
    ys_preds = [y_pred_1, y_pred_2, y_pred_3]
    return ys_preds

def sample_true_boxes():
    # (2, 1, 1, 1, 2, 4)
    true_boxes = np.load(os.path.join(TEST_SAMPLE_ROOT, "t_batch.npy")).astype(np.float32)
    return true_boxes

def sample_images():
    images = np.load(os.path.join(TEST_SAMPLE_ROOT, "x_batch.npy")).astype(np.float32)
    return images

