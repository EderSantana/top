"""
top.Optimizer accepts a generator as input to its run method. Here are some
examples of how to create generators to wrap datasets.
"""

import numpy as np

class Pylearn2DatasetIterator:
    """
    This iterator uses the new pylearn2 dataset.iterator interface

    Parameters
    ----------
    :param which_batches list: filter out batches from the pylearn2
        iterator, if this is equal to -1, it will keep all the batches.
    :param mode string: one of the original pylearn2 sequence modes like
        'sequential', 'shuffled_sequential', or a SubsetIterator object.
    """
    def __init__(self, dataset, batch_size,
                 which_batches=range(0,1), mode='shuffled_sequential'):
        self.dataset = dataset
        self.batch_size = batch_size
        self.mode = mode
        self.which_batches = which_batches

        self.pylean2_iterator = self.dataset.iterator(
                       batch_size,
                       self.dataset.get_num_samples(),
                       mode=self.mode,
                       data_specs=self.dataset.get_data_specs(),
                       return_tuple=True)
    def __iter__(self):
        return self

    def next(self):
        b = self.pylearn2_iterator.next()
        b = [b[i] for i in self.which_batches]
        return b

def Pylearn2OldGenerator(dataset,
                             batch_size,
                             mode='shuffled_sequential',
                             num_batches=-1):
    """
    This generator uses the old pylearn2 dataset.iterator interface
    """
    if num_batches == -1:
        num_batches = dataset.get_num_examples()/batch_size
    for b in dataset.iterator(
                     mode,
                     batch_size,
                     num_batches):
        yield b

def Pylearn2DatasetGenerator(dataset,
                             batch_size,
                             which_batches=range(0,1),
                             mode='shuffled_sequential',
                             num_batches=-1):
    """
    This generator uses the new pylearn2 dataset.iterator interface

    Parameters
    ----------
    :param which_batches list: filter out batches from the pylearn2
        iterator, if this is equal to -1, it will keep all the batches.
    :param mode string: one of the original pylearn2 sequence modes like
        'sequential', 'shuffled_sequential', or a SubsetIterator object.
    """

    for b in dataset.iterator(
                   batch_size,
                   dataset.get_num_examples()/batch_size,
                   mode=mode,
                   data_specs=dataset.get_data_specs(),
                   return_tuple=True
             ):
        b = [b[i] for i in which_batches]
        yield b

def GeneratorWithNoise(dataset,
                       batch_size,
                       noise_size,
                       which_batches=range(0,1),
                       mode='shuffled_sequential',
                       num_batches=-1):
    """
    This generator uses the new pylearn2 dataset.iterator interface.
    It also outputs some extra noisy values. Someties generating
    the noise using theano.rng is slower than transfering the data
    to the GPU. This happend to me, I don't know why.

    Parameters
    ----------
    :param which_batches list: filter out batches from the pylearn2
        iterator, if this is equal to -1, it will keep all the batches.
    :param mode string: one of the original pylearn2 sequence modes like
        'sequential', 'shuffled_sequential', or a SubsetIterator object.
    """

    for b in dataset.iterator(
                   batch_size,
                   dataset.get_num_examples()/batch_size,
                   mode=mode,
                   data_specs=dataset.get_data_specs(),
                   return_tuple=True
             ):
        b = [b[i] for i in which_batches]
        t, bsize, dim = b[0].shape # time length, batch size, dim
        eps = np.random.normal(0,1,size=(t, bsize, noise_size))
        b.append(eps)
        yield b

def NumpyDatasetGenerator(dataset,
                          batch_size,
                          shuffle=True,
                          num_batches=-1):
    """
    Feeds Numpy tensor batches.

    Parameters
    ----------
    :param shuffle bool: Shuffles the batches (axis=0) before starting the
                         generator.
    """
    # top.Optimizer is expecting for tuples
    if isinstance(dataset, tuple):
        dataset = tuple(dataset)

    if shuffle==True:
        perm = np.random.permutation(dataset[0].shape[0])
        dataset = [d[perm] for d in dataset]
    if num_batches == -1:
        num_batches = dataset[0].shape[0]/batch_size
    for i in range(num_batches):
        start  = i*batch_size
        finish = (i+1)*batch_size
        batch = [d[start:finish] for d in dataset]
        yield tuple(batch)
