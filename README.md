# CoordConv for Keras
Keras implementation of CoordConv from the paper [An intriguing failing of convolutional neural networks and the CoordConv solution](https://arxiv.org/abs/1807.03247).

Extends the `CoordinateChannel` concatenation from only 2D rank (images) to 1D (text / time series) and 3D tensors (video / voxels).

# Usage

Import `coord.py` and call it *before* any convolution layer in order to attach the coordinate channels to the input.

There are **3 different versions of CoordinateChannel** - 1D, 2D and 3D for each of `Conv1D`, `Conv2D` and `Conv3D`. 

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
The quadrant dataset model can be trained and evaluated in less than 25 epochs using `train_quadrant_classifier.py`

|Train | Test  |  Predictions  |
|:---: | :---: | :-----------: |
|<img src="https://github.com/titu1994/keras-coordconv/blob/master/images/quadrant-train.png?raw=true" > | <img src="https://github.com/titu1994/keras-coordconv/blob/master/images/quadrant-test.png?raw=true" > | <img src="https://github.com/titu1994/keras-coordconv/blob/master/images/quadrant-preds.png?raw=true" > |

# Checks

To see if the implementation of CoordConv index concatenation is correct, please refer to the numpy implementations in
the `checks` directory, for the implementation of all 3 versions.

## **Difference from paper**
This implementation of the coordinate channels creation differs slightly from the original paper.

The major difference is that for 2/3D Convolutions, it may not be the case that the height and width are the same
for all layers. The original implementation would throw an error due to shape mismatch during the concatenation.

To over come this, the `np.ones()` operation which occurs at the first of every channel is modified and a few
transpose operations are added to account for this change.

This modification along with some transpose operations allows for height and width to be different and still work.

## Theano Support

Theano is partially supported with the `coord_theano.py` script and using passing a static batch size to the Input layer.

# Requirements

- Keras 2.2.0+
- Either Tensorflow or CNTK backend.
- Matplotlib (to plot images only)
