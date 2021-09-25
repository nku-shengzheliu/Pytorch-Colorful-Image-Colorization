# Pytorch-Colorful Image Colorization

## Introduction

An unofficial pytorch implementation of "**Colorful image colorization**"

paper: https://link.springer.com/chapter/10.1007/978-3-319-46487-9_40

<img src="https://github.com/nku-shengzheliu/Pytorch-Colorful-Image-Colorization/blob/master/colornet.JPG" width = 60% height = 60% align=center/>

## Performance

I am running the paddlepaddle version of the model and the relevant paddle model weights file will be available here:

* [PaddlePaddle Colorful image colorization](https://github.com/nku-shengzheliu/PaddlePaddle-Colorful-Image-Colorization)

I don't have the free time and GPU to run this code these days... So I'm afraid I can't provide model weight files and performance reports for now. 

Anyway, if you encounter any problems during the training process, feel free to ask!

## Dataset

[ImageNet Dataset](https://image-net.org/download)  I **resized** all images to 256*256

- Training set：1281167 images
- Validation set：10000 images that from the imagenet validation set
- Test set：10000 images that from the imagenet validation set. Proposed by [Learning Representations for Automatic Colorization](http://people.cs.uchicago.edu/~larsson/colorization/)

## Installation

```
# clone this repo
git clone https://github.com/nku-shengzheliu/Pytorch-Colorful-Image-Colorization.git
cd Pytorch-Colorful-Image-Colorization
```

```
pip install -r requirements.txt
```

## Training

Train the model using the following commands:

```
python train.py
```

If training is interrupted, you can resume it with the `--resume` parameter, which sets `--resume` to the last saved weight file.

## Test and Visualization

```
python demo.py --data_path {data path} --split {train/val/test}  --model_path {saved_path/XXXX.pth} --save_path{path to save visualized results}
```

## Acknowledgement

Thanks for the work of [official project](https://github.com/richzhang/colorization) and [another pytorch implementation](https://github.com/Epiphqny/Colorization/tree/master/code). 









