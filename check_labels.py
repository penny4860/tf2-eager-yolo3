

import glob
import os

from yolo import PROJECT_ROOT
from yolo.dataset.annotation import PascalVocXmlParser

def get_unique_labels(files):
    parser = PascalVocXmlParser()
    labels = []
    for fname in files:
        labels += parser.get_labels(fname)
        labels = list(set(labels))
    labels.sort()
    return labels

if __name__ == '__main__':
    ann_root = os.path.join(os.path.dirname(PROJECT_ROOT), "dataset", "svhn", "voc_format_annotation", "train")
    
    # 1. create generator
    train_ann_fnames = glob.glob(os.path.join(ann_root, "*.xml"))
    print(get_unique_labels(train_ann_fnames))
    print(len(train_ann_fnames))


