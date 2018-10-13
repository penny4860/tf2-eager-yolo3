from setuptools import setup, find_packages

setup(name='tf-eager-yolov3',
      version=open("yolo/_version.py").readlines()[-1].split()[-1].strip("\"'"),
      description='object detection algorithm',
      author='jeongjoonsup',
      author_email='penny4860@gmail.com',
      # url='https://penny4860.github.io/',
      packages=find_packages(),
     )
