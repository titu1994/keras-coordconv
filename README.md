# CoordConv for Keras
Keras implementation of CoordConv from the paper [An intriguing failing of convolutional neural networks and the CoordConv solution](https://arxiv.org/abs/1807.03247).

Extends the `CoordinateChannel` concatenation from only 2D rank (images) to 1D (text / time series) and 3D tensors (video / voxels).

# Usage

Import `coord.py` and call it *before* any convolution layer in order to attach the coordinate channels to the input.

```python
from coord import CoordinateChannel2D

# prior to first conv
ip = Input(shape=(64, 64, 2))
x = CoordinateChannel2D()(ip)
x = Conv2D(...)(x)  # This defines the `CoordConv` from the paper.
...
x = CoordinateChannel2D(use_radius=True)(x)
x = Conv2D(...)(x)  # This adds the 3rd channel for the radius.
```

# Experiments

The experiments folder contains the `Classification` of a 64x64 grid using the coordinate index as input as in the paper for both `Uniform` and `Quadrant` datasets.

## Creating the datasets
First, edit the `make_dataset.py` file to change the `type` parameter - to either `uniform` or `quadrant`. This will generate 2 folders for the datasets and several numpy files.

## Uniform Dataset
The uniform dataset model can be trained and evaluated in less than 10 epochs using `train_uniform_classifier.py`.

|Train | Test  |  Predictions  |
|:---: | :---: | :-----------: |
|<img src="https://github.com/titu1994/keras-coordconv/blob/master/images/uniform-train.png?raw=true" > | <img src="https://github.com/titu1994/keras-coordconv/blob/master/images/uniform-test.png?raw=true" > | <img src="https://github.com/titu1994/keras-coordconv/blob/master/images/uniform-preds.png?raw=true" > |

## Quadrant Dataset
The uniform dataset model can be trained and evaluated in less than 25 epochs using `train_quadrant_classifier.py`

|Train | Test  |  Predictions  |
|:---: | :---: | :-----------: |
|<img src="https://github.com/titu1994/keras-coordconv/blob/master/images/quadrant-train.png?raw=true" > | <img src="https://github.com/titu1994/keras-coordconv/blob/master/images/quadrant-test.png?raw=true" > | <img src="https://github.com/titu1994/keras-coordconv/blob/master/images/quadrant-preds.png?raw=true" > |

# Requirements

- Keras 2.2.0+
- Either Tensorflow, Theano or CNTK backend.
- Matplotlib (to plot images only)
