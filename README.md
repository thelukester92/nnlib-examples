# nnlib-examples - Using the Neural Network Library

[nnlib](https://github.com/thelukester92/nnlib) is an all-header library for building, training, and using neural networks.
This repository is a collection of examples of how to use [nnlib](https://github.com/thelukester92/nnlib).

# Examples

To build all examples, simply run `make`.
To build a specific test, use `make bin/[test]` for optimized or `make bin/[test]_dbg` for debugging.
These tests are set up so that they can be used with data other than the data provided.
The tests currently available are...

* `classify` - train a classifier; by default, it uses the MNIST dataset.
* `timeseries` - train a RNN to extrapolate a time series; by default, it uses the Airline dataset.
