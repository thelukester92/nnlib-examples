# nnlib-examples - Using the Neural Network Library

[nnlib](https://github.com/thelukester92/nnlib) is an all-header library for building, training, and using neural networks.
This repository is a collection of examples of how to use [nnlib](https://github.com/thelukester92/nnlib).

# Examples

To build all examples in optimized mode, run `make` in the `src` folder.
To build all examples in debug mode, run `make dbg`.
To build a specific example in optimized mode, include the example name after `make`.
To build a specific example in debug mode, include the example name followed by `_dbg` after `make`.
All examples compile to an executable with the same name as the make target.

* `mnist` - train a classifier on the MNIST data set.
