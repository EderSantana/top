import itertools
import top
import theano
import theano.tensor as T
import numpy as np
from theano.compat.python2x import OrderedDict

class Optimizer():
  """Optimizer API
  """
  def __init__(self, parameters, cost, method='sgd',input=[], givens=None,
               constant=None,learning_rate=.001, momentum=None,
               lr_rate=None, m_rate=None, extra_updates=None,
               grad_clip=None):
    if not isinstance(parameters,list):
        parameters = [parameters]
    self.p = parameters

    self.cost = cost
    self.lr = theano.shared(np.asarray(learning_rate, dtype=theano.config.floatX))
    if momentum:
        self.m = theano.shared(np.asarray(momentum, dtype=theano.config.floatX))
    else:
        self.m = None
    self.input = input
    self.lr_rate = lr_rate
    self.m_rate = m_rate
    self.method = method
    self.givens = givens
    self.cc = constant
    self.extra_updates = extra_updates
    self.grad_clip = grad_clip

  def compile(self):
    print "$> Compiling optimizer."

    if self.method.lower() == 'sgd':
        updates = top.up.sgd(self.p, cost=self.cost, lr=self.lr,
                             momentum=self.m, lr_rate=self.lr_rate,
                             m_rate=self.m_rate, consider_cosntant=self.cc)
    elif self.method.lower() == 'rmsprop':
        updates = top.up.rmsprop(self.p,self.cost, lr=self.lr, momentum=self.m,
                                 lr_rate=self.lr_rate, m_rate=self.m_rate,
                                 consider_constant=self.cc,
                                 grad_clip=self.grad_clip)
    elif self.method.lower() == 'adam':
        updates = top.up.adam(self.p, self.cost, lr=self.lr,
                              grad_clip=self.grad_clip)
    elif self.method.lower() == 'adagrad':
        updates = top.up.adagrad(self.p, self.cost,
                                 lr=self.lr, lr_rate=self.lr_rate)
    else:
        raise NotImplementedError("Optimization method not implemented!")

    if self.extra_updates is not None:
        updates.append(self.extra_updates)

    updates = OrderedDict(updates)

    # This may seem weird, but I was getting bugs without this if-else
    if self.input == []:
        # Return cost and update params
        self.f = theano.function([], self.cost, updates=updates,
                                 givens=self.givens, allow_input_downcast=True)
        # Return cost without updating params, use this for testing
        self.g = theano.function([], self.cost, givens=self.givens,
                                 allow_input_downcast=True)
    else:
      if not isinstance(self.input,list):
          self.input = [self.input]
      self.f = theano.function(self.input, self.cost, updates=updates,
                               givens=self.givens, allow_input_downcast=True)
      self.g = theano.function(self.input, self.cost, givens=self.givens,
                               allow_input_downcast=True)

    return self

  def run(self,niter,*args):
    """:run: runs the Optimizer

    run extra argumets *args are numpy batches

    If the input is a dataset, use Optimizer.iterate.

    :param niter integer: number of passes over a given batch
    """
    total = 0.
    if not hasattr(self,'f'):
        self.compile()
    for k in range(niter):
        total += self.f(*args)

    return self, total

  def iterate(self, dataset):
      '''
      Similar to 'run' method but this method expects a datset as input
      '''
      if not hasattr(self,'f'):
          self.compile()
      total = 0.
      for N,b in enumerate(dataset):
          if not isinstance(b, tuple):
              b = tuple(b)
          total += self.f(*b)
      return total/N

  def iterate_epochs(self, nepochs, dataset):
      total = np.zeros(nepochs)
      for k in range(nepochs):
          data_copy,dataset = itertools.tee(dataset)
          total[k] = self.iterate(data_copy)
      return total
