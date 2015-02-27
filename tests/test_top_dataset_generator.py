from pylearn2.datasets.mnist import MNIST
from top.dataset_iterator import Pylearn2DatasetGenerator, Pylearn2OldGenerator

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

if __name__=='__main__':
    test_pylearn2_generator()
