import numpy as np

class Pylearn2DatasetIterator:
    """
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

    for b in dataset.iterator(
                   batch_size,
                   dataset.get_num_examples()/batch_size,
                   mode=mode,
                   data_specs=dataset.get_data_specs(),
                   return_tuple=True
             ):
        b = [b[i] for i in which_batches]
        yield b
