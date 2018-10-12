

import numpy as np
import cv2

from utils.image import apply_random_scale_and_crop, random_flip, correct_bounding_boxes


def _aug_image(instance, net_h, net_w, jitter):
    image_name = instance['filename']
    image = cv2.imread(image_name) # RGB image
    
    if image is None: print('Cannot find ', image_name)
    image = image[:,:,::-1] # RGB image
        
    image_h, image_w, _ = image.shape
    
    # determine the amount of scaling and cropping
    dw = jitter * image_w;
    dh = jitter * image_h;

    new_ar = (image_w + np.random.uniform(-dw, dw)) / (image_h + np.random.uniform(-dh, dh));
    scale = np.random.uniform(0.25, 2);

    if (new_ar < 1):
        new_h = int(scale * net_h);
        new_w = int(net_h * new_ar);
    else:
        new_w = int(scale * net_w);
        new_h = int(net_w / new_ar);
        
    dx = int(np.random.uniform(0, net_w - new_w));
    dy = int(np.random.uniform(0, net_h - new_h));
    
    new_w, new_h = net_w, net_h
    dx = 0
    dy = 0
    
    # apply scaling and cropping
    im_sized = apply_random_scale_and_crop(image, new_w, new_h, net_w, net_h, dx, dy)
    
    # randomly distort hsv space
    # im_sized = random_distort_image(im_sized)
    
    # randomly flip
    flip = np.random.randint(2)
    
    flip = 0
    im_sized = random_flip(im_sized, flip)
        
    # correct the size and pos of bounding boxes
    all_objs = correct_bounding_boxes(instance['object'], new_w, new_h, net_w, net_h, dx, dy, flip, image_w, image_h)
    
    return im_sized, all_objs   

