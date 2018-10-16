# -*- coding: utf-8 -*-

import numpy as np
from keras.utils import Sequence
from yolo.dataset.augment import ImgAugment
from yolo.utils.box import create_anchor_boxes

# ratio between network input's size and network output's size, 32 for YOLOv3
DOWNSAMPLE_RATIO = 32
DEFAULT_NETWORK_SIZE = 288


class BatchGenerator(Sequence):
    def __init__(self, 
        annotations, 
        anchors,   
        max_box_per_image=30,
        batch_size=2,
        min_net_size=320,
        max_net_size=608,    
        shuffle=True, 
        jitter=True, 
    ):
        self.annotations          = annotations
        self._batch_size         = batch_size
        self.max_box_per_image  = max_box_per_image
        self.min_net_size       = (min_net_size//DOWNSAMPLE_RATIO)*DOWNSAMPLE_RATIO
        self.max_net_size       = (max_net_size//DOWNSAMPLE_RATIO)*DOWNSAMPLE_RATIO
        self.shuffle            = shuffle
        self.jitter             = jitter
        self.anchors            = create_anchor_boxes(anchors)
        self.net_size = DEFAULT_NETWORK_SIZE

        if shuffle: np.random.shuffle(self.annotations)
            
    def __len__(self):
        return int(np.ceil(float(len(self.annotations))/self._batch_size))           

    def __getitem__(self, idx):
        # get image input size, change every 10 batches
        net_size = self._get_net_size(idx)
        base_grid_h, base_grid_w = net_size//DOWNSAMPLE_RATIO, net_size//DOWNSAMPLE_RATIO

        # determine the first and the last indices of the batch
        x_batch = []
        t_batch = np.zeros((self._batch_size, 1, 1, 1,  self.max_box_per_image, 4))   # list of groundtruth boxes

        # initialize the inputs and the outputs
        n_classes = self.annotations.n_classes()
        yolo_1 = np.zeros((self._batch_size, 1*base_grid_h,  1*base_grid_w, len(self.anchors)//3, 4+1+n_classes)) # desired network output 1
        yolo_2 = np.zeros((self._batch_size, 2*base_grid_h,  2*base_grid_w, len(self.anchors)//3, 4+1+n_classes)) # desired network output 2
        yolo_3 = np.zeros((self._batch_size, 4*base_grid_h,  4*base_grid_w, len(self.anchors)//3, 4+1+n_classes)) # desired network output 3
        yolos = [yolo_3, yolo_2, yolo_1]

        true_box_index = 0

        for i in range(self._batch_size):
            # 1. get input file & its annotation
            fname = self.annotations.fname(self._batch_size*idx + i)
            boxes = self.annotations.boxes(self._batch_size*idx + i)
            labels = self.annotations.code_labels(self._batch_size*idx + i)

            # 2. read image in fixed size
            img_augmenter = ImgAugment(net_size, net_size, False)
            img, boxes = img_augmenter.imread(fname, boxes)
            
            x_batch.append(normalize(img))

            for original_box, label in zip(boxes, labels):
                max_anchor, scale_index, box_index = find_match_anchor(original_box, self.anchors)
                
                yolobox = yolo_box(yolos[scale_index], original_box, max_anchor, net_size, net_size)
                assign_box(yolos[scale_index][i], box_index, yolobox, label)

                # assign the true box to t_batch
                t_batch[i, 0, 0, 0, true_box_index] = true_box(yolos[scale_index], original_box, net_size, net_size)

                true_box_index += 1
                true_box_index  = true_box_index % self.max_box_per_image    

        return np.array(x_batch), t_batch, yolo_1, yolo_2, yolo_3

    def _get_net_size(self, idx):
        if idx%10 == 0:
            net_size = DOWNSAMPLE_RATIO*np.random.randint(self.min_net_size/DOWNSAMPLE_RATIO, \
                                                         self.max_net_size/DOWNSAMPLE_RATIO+1)
            print("resizing: ", net_size, net_size)
            self.net_size = net_size
        return self.net_size

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.annotations)


def true_box(yolo, original_box, net_w, net_h):
    x1, y1, x2, y2 = original_box
    
    # determine the yolo to be responsible for this bounding box
    grid_h, grid_w = yolo.shape[1:3]
    
    # determine the position of the bounding box on the grid
    center_x = .5*(x1 + x2)
    center_x = center_x / float(net_w) * grid_w # sigma(t_x) + c_x
    center_y = .5*(y1 + y2)
    center_y = center_y / float(net_h) * grid_h # sigma(t_y) + c_y
    
    true_box = [center_x, center_y, x2 - x1, y2 - y1]
    return true_box


def yolo_box(yolo, original_box, anchor_box, net_w, net_h):
    
    x1, y1, x2, y2 = original_box
    _, _, anchor_w, anchor_h = anchor_box
    
    # determine the yolo to be responsible for this bounding box
    grid_h, grid_w = yolo.shape[1:3]
    
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


def find_match_anchor(box, anchor_boxes):
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


def assign_box(yolo, box_index, box, label):
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


import os
from yolo import PROJECT_ROOT
def create_generator(image_dir, annotation_dir):
    from yolo.dataset.annotation import parse_annotation
    train_anns = parse_annotation(annotation_dir,
                                  image_dir,
                                  labels_naming=["raccoon"])
    generator = BatchGenerator(train_anns,
                               anchors=[17,18, 28,24, 36,34, 42,44, 56,51, 72,66, 90,95, 92,154, 139,281],
                               min_net_size=288,
                               max_net_size=288,
                               shuffle=False)
    return generator


if __name__ == '__main__':
    def test(x_batch, t_batch, yolo_1, yolo_2, yolo_3):
        expected_x_batch = np.load(os.path.join(PROJECT_ROOT, "samples//x_batch.npy"))
        expected_t_batch = np.load(os.path.join(PROJECT_ROOT, "samples//t_batch.npy"))
        expected_yolo_1 = np.load(os.path.join(PROJECT_ROOT, "samples//yolo_1.npy"))
        expected_yolo_2 = np.load(os.path.join(PROJECT_ROOT, "samples//yolo_2.npy"))
        expected_yolo_3 = np.load(os.path.join(PROJECT_ROOT, "samples//yolo_3.npy"))
        
        for a, b in zip([x_batch, t_batch, yolo_1, yolo_2, yolo_3],
                        [expected_x_batch, expected_t_batch, expected_yolo_1, expected_yolo_2, expected_yolo_3]):
            if np.allclose(a, b):
                print("Test Passed")
            else:
                print("Test Failed")

    ann_dir = os.path.join(PROJECT_ROOT, "samples", "anns")
    img_dir = os.path.join(PROJECT_ROOT, "samples", "imgs")
    generator = create_generator(img_dir, ann_dir)
    x_batch, t_batch, yolo_1, yolo_2, yolo_3 = generator[0]
    test(x_batch, t_batch, yolo_1, yolo_2, yolo_3)
    

