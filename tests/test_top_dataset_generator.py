import top
import numpy as np
import theano

from theano import tensor
from pylearn2.datasets.mnist import MNIST
from top.dataset_iterator import (Pylearn2DatasetGenerator,
                                  Pylearn2OldGenerator,
                                  NumpyDatasetGenerator)

def test_pylearn2_old_generator():
    dataset = MNIST('train')
    dataset_generator = Pylearn2OldGenerator(
                            dataset,
                            batch_size=1000,
                            num_batches=-1
                            )
    for b in dataset_generator:
        print b.shape
        assert b.shape == (1000,784)

def test_pylearn2_generator():
    dataset = MNIST('train')
    dataset_generator = Pylearn2OldGenerator(
                            dataset,
                            batch_size=1000,
                            num_batches=-1
                            )
    for b in dataset_generator:
        print b.shape
        assert b.shape == (1000,784)

def test_numpy_generator():
    dataset = np.random.normal(0,1,(1000,10))
    target  = 0. + (np.dot(dataset, np.ones((10,1)))>.5)
    dataset_generator = NumpyDatasetGenerator(
                          dataset=(dataset, target),
                          batch_size=100
                          )
    for b in dataset_generator:
        assert b[0].shape == (100,10)
        assert b[1].shape == (100,1)

def test_top_with_numpy_generator():
    dataset = np.random.normal(0,1,(1000,10))
    target  = 0. + (np.dot(dataset, np.ones((10,1)))>.5)
    dataset_generator = NumpyDatasetGenerator(
                          dataset=(dataset, target),
                          batch_size=100
                          )
    W  = theano.shared(np.random.normal(0,1,(10,1)).astype(top.up.floatX))
    X, T = tensor.matrices('X', 'Y')
    Y = tensor.nnet.sigmoid(tensor.dot(X,W))
    cost = tensor.nnet.binary_crossentropy(Y, T).mean()
    opt = top.Optimizer(W, cost, input=[X,T], method='sgd', learning_rate=.01)
    opt.run_epochs(1000, dataset_generator)
    print W.get_value()

if __name__=='__main__':
    #test_pylearn2_generator()
    test_top_with_numpy_generator()
