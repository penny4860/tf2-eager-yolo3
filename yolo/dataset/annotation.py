# -*- coding: utf-8 -*-

import os
import numpy as np
from xml.etree.ElementTree import parse


def parse_annotation(ann_fname, img_dir, labels_naming=[]):
    # Todo : labels_naming 이 없으면 모든 labels을 자동으로 parsing
    parser = PascalVocXmlParser()
    
    fname = parser.get_fname(ann_fname)

    annotation = Annotation(os.path.join(img_dir, fname))

    labels = parser.get_labels(ann_fname)
    boxes = parser.get_boxes(ann_fname)
    
    for label, box in zip(labels, boxes):
        x1, y1, x2, y2 = box
        if label in labels_naming:
            annotation.add_object(x1, y1, x2, y2, name=label, code=labels_naming.index(label))
    return annotation.fname, annotation.boxes, annotation.coded_labels


def get_unique_labels(files):
    parser = PascalVocXmlParser()
    labels = []
    for fname in files:
        labels += parser.get_labels(fname)
        labels = list(set(labels))
    labels.sort()
    return labels


class PascalVocXmlParser(object):
    """Parse annotation for 1-annotation file """
    
    def __init__(self):
        pass

    def get_fname(self, annotation_file):
        """
        # Args
            annotation_file : str
                annotation file including directory path
        
        # Returns
            filename : str
        """
        root = self._root_tag(annotation_file)
        return root.find("filename").text

    def get_width(self, annotation_file):
        """
        # Args
            annotation_file : str
                annotation file including directory path
        
        # Returns
            width : int
        """
        tree = self._tree(annotation_file)
        for elem in tree.iter():
            if 'width' in elem.tag:
                return int(elem.text)

    def get_height(self, annotation_file):
        """
        # Args
            annotation_file : str
                annotation file including directory path
        
        # Returns
            height : int
        """
        tree = self._tree(annotation_file)
        for elem in tree.iter():
            if 'height' in elem.tag:
                return int(elem.text)

    def get_labels(self, annotation_file):
        """
        # Args
            annotation_file : str
                annotation file including directory path
        
        # Returns
            labels : list of strs
        """

        root = self._root_tag(annotation_file)
        labels = []
        obj_tags = root.findall("object")
        for t in obj_tags:
            labels.append(t.find("name").text)
        return labels
    
    def get_boxes(self, annotation_file):
        """
        # Args
            annotation_file : str
                annotation file including directory path
        
        # Returns
            bbs : 2d-array, shape of (N, 4)
                (x1, y1, x2, y2)-ordered
        """
        root = self._root_tag(annotation_file)
        bbs = []
        obj_tags = root.findall("object")
        for t in obj_tags:
            box_tag = t.find("bndbox")
            x1 = box_tag.find("xmin").text
            y1 = box_tag.find("ymin").text
            x2 = box_tag.find("xmax").text
            y2 = box_tag.find("ymax").text
            box = np.array([int(float(x1)), int(float(y1)), int(float(x2)), int(float(y2))])
            bbs.append(box)
        bbs = np.array(bbs)
        return bbs

    def _root_tag(self, fname):
        tree = parse(fname)
        root = tree.getroot()
        return root

    def _tree(self, fname):
        tree = parse(fname)
        return tree


class Annotation(object):
    """
    # Attributes
        fname : image file path
        labels : list of strings
        boxes : Boxes instance
    """
    def __init__(self, filename):
        self.fname = filename
        self.labels = []
        self.coded_labels = []
        self.boxes = None

    def add_object(self, x1, y1, x2, y2, name, code):
        self.labels.append(name)
        self.coded_labels.append(code)
        
        if self.boxes is None:
            self.boxes = np.array([x1, y1, x2, y2]).reshape(-1,4)
        else:
            box = np.array([x1, y1, x2, y2]).reshape(-1,4)
            self.boxes = np.concatenate([self.boxes, box])


if __name__ == '__main__':
    from yolo import PROJECT_ROOT
    ann_dir = os.path.join(PROJECT_ROOT, "samples", "anns") + "//"
    img_dir = os.path.join(PROJECT_ROOT, "samples", "imgs") + "//"
    train_anns = parse_annotation(ann_dir,
                                  img_dir,
                                  labels_naming=["raccoon"])

    ann = train_anns[0]
    print(ann.fname, ann.labels, ann.boxes)








