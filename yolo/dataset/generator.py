# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

from yolo.dataset.augment import ImgAugment
from yolo.utils.box import create_anchor_boxes
from yolo.dataset.annotation import parse_one_ann
from yolo import COCO_ANCHORS

# ratio between network input's size and network output's size, 32 for YOLOv3
DOWNSAMPLE_RATIO = 32
DEFAULT_NETWORK_SIZE = 288


def create_generator(image_dir,
                     annotation_dir,
                     batch_size,
                     labels_naming=["raccoon"],
                     anchors=COCO_ANCHORS,
                     min_net_size=288,
                     max_net_size=288,
                     shuffle=True,
                     jitter=True):
    """
    # Args
        image_dir : str
        annotation_dir : str
        labels_naming : list of strs
        anchors : list of integer (length 18)
    # Returns
        generator : tensorflow.keras.utils.Sequence
            generator[0] -> xs, ys_1, ys_2, ys_3
    """
    import glob
    import os
    ann_fnames = glob.glob(os.path.join(annotation_dir, "*.xml"))
    print(ann_fnames)
    generator = BatchGenerator(ann_fnames, image_dir, labels=["raccoon"], anchors=anchors, min_net_size=min_net_size,
                               jitter=False)
    
    def gen():
        i = -1
        while True:
            i += 1
            yield generator.get(i)
    
    n_features = len(labels_naming) + 4 + 1
    ds = tf.data.Dataset.from_generator(gen,
                                        (tf.float32, tf.float32, tf.float32, tf.float32),
                                        (tf.TensorShape([None, None, 3]),
                                         tf.TensorShape([None, None, 3, n_features]),
                                         tf.TensorShape([None, None, 3, n_features]),
                                         tf.TensorShape([None, None, 3, n_features])))
    ds = ds.batch(batch_size)
    # ds = ds.shuffle(buffer_size=256, reshuffle_each_iteration=shuffle)
    # Todo : shuffle 적용
    # multi-scale generate
    iterator = ds.make_one_shot_iterator()
    return iterator


# ann_fnames = glob.glob(ann_dir+"/*.xml")
# img_dir = os.path.join(PROJECT_ROOT, "tests", "dataset", "raccoon", "imgs")


class BatchGenerator(object):
    def __init__(self, 
                 ann_fnames,
                 img_dir,
                 labels,
                 anchors,   
                 min_net_size=320,
                 max_net_size=608,    
                 jitter=True):

        self.ann_fnames = ann_fnames
        self.img_dir = img_dir
        self.lable_names = labels
        self.min_net_size       = (min_net_size//DOWNSAMPLE_RATIO)*DOWNSAMPLE_RATIO
        self.max_net_size       = (max_net_size//DOWNSAMPLE_RATIO)*DOWNSAMPLE_RATIO
        self.jitter             = jitter
        self.anchors            = create_anchor_boxes(anchors)
        self.net_size = DEFAULT_NETWORK_SIZE

    def get(self, i):

        index = i % len(self.ann_fnames)        
        fname, boxes, coded_labels = parse_one_ann(self.ann_fnames[index], self.img_dir, self.lable_names)

        
        # net_size = self._get_net_size(idx)
        net_size = self.min_net_size
        
        list_ys = _create_empty_xy(net_size, len(self.lable_names))

        # 1. get input file & its annotation

        # 2. read image in fixed size
        img_augmenter = ImgAugment(net_size, net_size, self.jitter)
        img, boxes_ = img_augmenter.imread(fname, boxes)

        # 4. Append ys
        for original_box, label in zip(boxes_, coded_labels):
            max_anchor, scale_index, box_index = _find_match_anchor(original_box, self.anchors)
            
            _coded_box = _encode_box(list_ys[scale_index], original_box, max_anchor, net_size, net_size)
            _assign_box(list_ys[scale_index], box_index, _coded_box, label)

        return normalize(img), list_ys[2], list_ys[1], list_ys[0]


def _create_empty_xy(net_size, n_classes, n_boxes=3):
    # get image input size, change every 10 batches
    base_grid_h, base_grid_w = net_size//DOWNSAMPLE_RATIO, net_size//DOWNSAMPLE_RATIO

    # initialize the inputs and the outputs
    ys_1 = np.zeros((1*base_grid_h,  1*base_grid_w, n_boxes, 4+1+n_classes)) # desired network output 1
    ys_2 = np.zeros((2*base_grid_h,  2*base_grid_w, n_boxes, 4+1+n_classes)) # desired network output 2
    ys_3 = np.zeros((4*base_grid_h,  4*base_grid_w, n_boxes, 4+1+n_classes)) # desired network output 3
    list_ys = [ys_3, ys_2, ys_1]
    return list_ys


def _encode_box(yolo, original_box, anchor_box, net_w, net_h):
    
    x1, y1, x2, y2 = original_box
    _, _, anchor_w, anchor_h = anchor_box
    
    # determine the yolo to be responsible for this bounding box
    grid_h, grid_w = yolo.shape[:2]
    
    # determine the position of the bounding box on the grid
    center_x = .5*(x1 + x2)
    center_x = center_x / float(net_w) * grid_w # sigma(t_x) + c_x
    center_y = .5*(y1 + y2)
    center_y = center_y / float(net_h) * grid_h # sigma(t_y) + c_y
    
    # determine the sizes of the bounding box
    w = np.log((x2 - x1) / float(anchor_w)) # t_w
    h = np.log((y2 - y1) / float(anchor_h)) # t_h

    box = [center_x, center_y, w, h]
    return box


def _find_match_anchor(box, anchor_boxes):
    """
    # Args
        box : array, shape of (4,)
        anchor_boxes : array, shape of (9, 4)
    """
    from yolo.utils.box import find_match_box
    x1, y1, x2, y2 = box
    shifted_box = np.array([0, 0, x2-x1, y2-y1])

    max_index = find_match_box(shifted_box, anchor_boxes)
    max_anchor = anchor_boxes[max_index]

    scale_index = max_index // 3
    box_index = max_index%3
    return max_anchor, scale_index, box_index


def _assign_box(yolo, box_index, box, label):
    center_x, center_y, _, _ = box

    # determine the location of the cell responsible for this object
    grid_x = int(np.floor(center_x))
    grid_y = int(np.floor(center_y))

    # assign ground truth x, y, w, h, confidence and class probs to y_batch
    yolo[grid_y, grid_x, box_index]      = 0
    yolo[grid_y, grid_x, box_index, 0:4] = box
    yolo[grid_y, grid_x, box_index, 4  ] = 1.
    yolo[grid_y, grid_x, box_index, 5+label] = 1


def normalize(image):
    return image/255.


if __name__ == '__main__':
    tf.enable_eager_execution()
    import os
    from yolo import PROJECT_ROOT
    def test(x_batch, yolo_1, yolo_2, yolo_3):
        expected_x_batch = np.load("x_batch.npy")
        expected_yolo_1 = np.load("yolo_1.npy")
        expected_yolo_2 = np.load("yolo_2.npy")
        expected_yolo_3 = np.load("yolo_3.npy")
        
        for a, b in zip([x_batch, yolo_1, yolo_2, yolo_3],
                        [expected_x_batch, expected_yolo_1, expected_yolo_2, expected_yolo_3]):
            if np.allclose(a, b):
                print("Test Passed")
            else:
                print("Test Failed")
                print(np.sum(a-b))

    ann_dir = os.path.join(PROJECT_ROOT, "tests", "dataset", "raccoon", "anns")
    img_dir = os.path.join(PROJECT_ROOT, "tests", "dataset", "raccoon", "imgs")

    iterator = create_generator(img_dir, ann_dir, 2)
    test(*iterator.get_next())

