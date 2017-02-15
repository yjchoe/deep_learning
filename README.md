# deep_learning

Python implementations of feedforward neural networks, restricted Boltzmann machines, deep Boltzmann machines, and autoencoders from scratch. For (much) more scalable implementations of these models, see standard deep learning libraries such as Theano and TensorFlow. 

This project was done as part of [this course](http://www.cs.cmu.edu/~rsalakhu/10807_2016/).

## Dependencies

The only dependencies outside of the Python standard library are:

* `scipy.special.expit`
* `sklearn.preprocessing.StandardScaler`

Both of these functions can easily be replaced with NumPy functions.

## Directory description

* `code` directory contains the package `NN`, which includes implementations of 
    * basic feedforward neural networks (`code/nn/`),
    * restricted Boltzmann machines (`code/rbm.py`), 
    * deep Boltzmann machines (`code/dbm.py`), and
    * autoencoders and denoising autoencoders (`code/autoencoder.py`).

* Scripts and analyses of these neural networks can be found in the `notebooks` directory. 
