# -*- coding: utf-8 -*-

import numpy as np
from keras.utils import Sequence
from utils.bbox import BoundBox, bbox_iou

from yolo.dataset.augment import ImgAugment

class BatchGenerator(Sequence):
    def __init__(self, 
        instances, 
        anchors,   
        labels,        
        downsample=32, # ratio between network input's size and network output's size, 32 for YOLOv3
        max_box_per_image=30,
        batch_size=1,
        min_net_size=320,
        max_net_size=608,    
        shuffle=True, 
        jitter=True, 
        norm=None
    ):
        self.annotations          = instances
        self._batch_size         = batch_size
        self.labels             = labels
        self.downsample         = downsample
        self.max_box_per_image  = max_box_per_image
        self.min_net_size       = (min_net_size//self.downsample)*self.downsample
        self.max_net_size       = (max_net_size//self.downsample)*self.downsample
        self.shuffle            = shuffle
        self.jitter             = jitter
        self.norm               = norm
        self.anchors            = [BoundBox(0, 0, anchors[2*i], anchors[2*i+1]) for i in range(len(anchors)//2)]
        self.net_h              = 416  
        self.net_w              = 416

        if shuffle: np.random.shuffle(self.annotations)
            
    def __len__(self):
        return int(np.ceil(float(len(self.annotations))/self._batch_size))           

    def __getitem__(self, idx):
        # get image input size, change every 10 batches
        net_h, net_w = self._get_net_size(idx)
        base_grid_h, base_grid_w = net_h//self.downsample, net_w//self.downsample

        # determine the first and the last indices of the batch
        x_batch = []
        t_batch = np.zeros((self._batch_size, 1, 1, 1,  self.max_box_per_image, 4))   # list of groundtruth boxes

        # initialize the inputs and the outputs
        yolo_1 = np.zeros((self._batch_size, 1*base_grid_h,  1*base_grid_w, len(self.anchors)//3, 4+1+len(self.labels))) # desired network output 1
        yolo_2 = np.zeros((self._batch_size, 2*base_grid_h,  2*base_grid_w, len(self.anchors)//3, 4+1+len(self.labels))) # desired network output 2
        yolo_3 = np.zeros((self._batch_size, 4*base_grid_h,  4*base_grid_w, len(self.anchors)//3, 4+1+len(self.labels))) # desired network output 3
        yolos = [yolo_3, yolo_2, yolo_1]

        true_box_index = 0

        for i in range(self._batch_size):
            # 1. get input file & its annotation
            fname = self.annotations.fname(self._batch_size*idx + i)
            boxes = self.annotations.boxes(self._batch_size*idx + i)
            labels = self.annotations.code_labels(self._batch_size*idx + i)

            # 2. read image in fixed size
            img_augmenter = ImgAugment(net_w, net_h, False)
            img, boxes = img_augmenter.imread(fname, boxes)
            
            x_batch.append(self.norm(img))

            for original_box, label in zip(boxes, labels):
                max_anchor, scale_index, box_index = find_match_anchor(original_box, self.anchors)
                
                yolobox = yolo_box(yolos[scale_index], original_box, max_anchor, net_w, net_h)
                assign_box(yolos[scale_index][i], box_index, yolobox, label)

                # assign the true box to t_batch
                t_batch[i, 0, 0, 0, true_box_index] = true_box(original_box, yolobox)

                true_box_index += 1
                true_box_index  = true_box_index % self.max_box_per_image    

        return np.array(x_batch), t_batch, yolo_1, yolo_2, yolo_3

    def _get_net_size(self, idx):
        if idx%10 == 0:
            net_size = self.downsample*np.random.randint(self.min_net_size/self.downsample, \
                                                         self.max_net_size/self.downsample+1)
            print("resizing: ", net_size, net_size)
            self.net_h, self.net_w = net_size, net_size
        return self.net_h, self.net_w

    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.annotations)


def true_box(box, yolobox):
    x1, y1, x2, y2 = box
    center_x, center_y, _, _ = yolobox
    true_box = [center_x, center_y, x2 - x1, y2 - y1]
    return true_box


def yolo_box(yolo, box, anchor, net_w, net_h):
    
    x1, y1, x2, y2 = box
    
    # determine the yolo to be responsible for this bounding box
    grid_h, grid_w = yolo.shape[1:3]
    
    # determine the position of the bounding box on the grid
    center_x = .5*(x1 + x2)
    center_x = center_x / float(net_w) * grid_w # sigma(t_x) + c_x
    center_y = .5*(y1 + y2)
    center_y = center_y / float(net_h) * grid_h # sigma(t_y) + c_y
    
    # determine the sizes of the bounding box
    w = np.log((x2 - x1) / float(anchor.xmax)) # t_w
    h = np.log((y2 - y1) / float(anchor.ymax)) # t_h

    box = [center_x, center_y, w, h]
    return box


def find_match_anchor(box, anchors):
    """
    # Args
        box : array, shape of (4,)
        anchors : list of BoundBox (9)
    """
    x1, y1, x2, y2 = box
    
    max_anchor = None                
    max_index  = -1
    max_iou    = -1
    
    shifted_box = BoundBox(0, 
                           0,
                           x2-x1,                                                
                           y2-y1)    
    
    for i in range(len(anchors)):
        anchor = anchors[i]
        iou    = bbox_iou(shifted_box, anchor)

        if max_iou < iou:
            max_anchor = anchor
            max_index  = i
            max_iou    = iou

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

    import os
    from yolo.dataset.annotation import parse_annotation
    from yolo import PROJECT_ROOT
    from utils.utils import normalize
    ann_dir = os.path.join(PROJECT_ROOT, "samples", "anns")
    img_dir = os.path.join(PROJECT_ROOT, "samples", "imgs")
    train_anns = parse_annotation(ann_dir,
                                  img_dir,
                                  labels_naming=["raccoon"])
    generator = BatchGenerator(train_anns,
                               anchors=[17,18, 28,24, 36,34, 42,44, 56,51, 72,66, 90,95, 92,154, 139,281],
                               min_net_size=288,
                               max_net_size=288,
                               shuffle=False,
                               norm=normalize,
                               labels=["raccoon"])
    x_batch, t_batch, yolo_1, yolo_2, yolo_3 = generator[0]
     
    test(x_batch, t_batch, yolo_1, yolo_2, yolo_3)
    

