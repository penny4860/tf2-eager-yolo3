
# Yolo-v3 using tensorflow eager

I have implemented yolo-v3 detector using tensorflow eager execution.

<img src="imgs/sample_detected.png" height="600">

## Usage for python code

#### 0. Requirement

* python 3.6
* anaconda 5.1.0
* tensorflow 1.8.0
* opencv 3.3.0
* imgaug
* Etc.

I recommend that you create and use an anaconda env that is independent of your project. You can create anaconda env for this project by following these simple steps. This process has been verified on Windows 10 and ubuntu 16.04.

```
$ conda create -n yolo3 python=3.6 anaconda=5.1.0
$ activate yolo3 # in linux "source activate yolo3"
(yolo) $ pip install tensorflow==1.8.0
(yolo) $ pip install opencv-python
(yolo) $ pip install imgaug
(yolo) $ pip install -e .
```

### 1. Object detection using original yolo3-weights

* Download the pretrained weight file from [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights).

* Run object detection through the following command.
	* ```project/root> python pred.py -w yolov3.weights -i imgs/sample.jpeg```

* You can see the following results:
	* <img src="imgs/dog_detected.jpeg" height="600">

### 2. Training from scratch

This project provides a way to train digit detector from scratch. If you follow the command below, you can build a digit detector with just two images. Please refer to the [train_from_scratch.ipynb](https://github.com/penny4860/tf-eager-yolo3/tree/master/examples/train_from_scratch.ipynb). If you follow the instructions, you can train the digit detector as shown below.

<img src="imgs/svhn.jpg" height="250">

Now you can add more images to train a digit detector with good generalization performance.

## Copyright

* See [LICENSE](LICENSE) for details.

