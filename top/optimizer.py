import cPickle
import itertools
import top
import theano
import theano.tensor as T
import numpy as np
from theano.compat.python2x import OrderedDict
import matplotlib.pyplot as plt
import time
from copy import copy

bokeh = None
try:
    import bokeh
    import bokeh.plotting as bplt
except:
    import warnings
    warnings.warn('Problem loading Bokeh, we can only use matplotlib for plotting.')
    bokeh=None

class Optimizer():
  """Optimizer API
  
  Basic usage
  ----------
  opt = top.Optimizer(params, cost, method='sgd', input=[X])
  opt.run(num_epochs, input_data)

  Parameters
  ---------
  parameters: list
      list of theano tensors
  cost: theano.scalar 
      theano scalar theano expression
  method: str
      a valid optimization method: 'sgd', 'rmsprop', 'adam', 'adagrad'
  input: list
      list of theano tensors used to calculate cost
  givens: dict
  lr_rate: fload
      rate change per iteration of the learning rate: lr *= lr_rate 
  momentum: float
      rate change per iteration of the momentum: m *= m_rate
  extra_updatas: OrderedDict
  grad_clip: float
      maximum norm for the gradients, valid input for 'rmsprop' and 'adam'
  ipython_display: IPython.display
  bokeh_server: str
      address for a bokeh server
  """
  def __init__(self, parameters, cost, method='sgd',input=[], givens=None,
               constant=None,learning_rate=.001, momentum=None,
               lr_rate=None, m_rate=None, extra_updates=None,
               grad_clip=None, ipython_display=None, bokeh_server=None):

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
    self.ipython_display = ipython_display # image logging of cost function

    self.bokeh_server = bokeh_server
    if bokeh_server is not None:
        bplt.output_notebook(url=bokeh_server)

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

    updates = OrderedDict(updates)
    if self.extra_updates is not None:
        updates.update(self.extra_updates)

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
      N = 0.
      for b in dataset():
          if not isinstance(b, tuple):
              b = tuple(b)
          total += self.f(*b)
          N = N+1.

      if self.bokeh_server is not None:
          self.bokeh_plotting(total/N)
      if self.ipython_display is not None:
          self.mpl_plotting(total/N)

      return total/N

  def testiterate(self, testset):
      '''
      Similar to iterate but does not update parameters
      '''
      if not hasattr(self, 'g'):
          self.compile()
      testtotal = 0.
      N = 0.
      for b in testset():
          if not isinstance(b, tuple):
              b = tuple(b)
          testtotal += self.g(*b)
          N += 1.
      return testtotal/N

  def iterate_epochs(self, nepochs, dataset):
      total = [] #np.zeros(nepochs)
      for k in range(nepochs):
          total.append(self.iterate( dataset ))
      return total

  def train_valid_save(self, nepochs, trainset, validset, what_to_save,
                       where_to_save, save_every=1):
      self.total = []
      self.validtotal = []
      for k in range(nepochs):
          if k % save_every == 0:
              self.validtotal = self.valid_save(self.validtotal,
                                 validset, what_to_save, where_to_save)
          self.total.append(self.iterate(trainset))
          yield self
      #return total, validtotal

  def valid_save(self, validtotal, validset, what_to_save, where_to_save):

      validtotal.append( self.testiterate( validset ) )
      if validtotal[-1] == np.min(validtotal):
          # log saving best model
          print "Saving model with validation cost %f" % validtotal[-1]
          cPickle.dump(what_to_save, file(where_to_save, 'w'), -1)
      return validtotal

  def mpl_plotting(self, total):
      if self.ipython_display is not None:
          plt.cla()
          plt.plot(total)
          self.ipython_display.clear_output(wait=True)
          self.ipython_display.display(plt.gcf())
          #time.sleep(0.1)

  def bokeh_plotting(self, total):
      if self.bokeh_server is not None:
          x = range(len(total))

          if not hasattr(self, 'fig'):
              self.fig = bplt.figure(title='top.Optimizer')
              self.fig.line(x, total, legend='cost', x_axis_label='epoch number',
                            name='top_figure', plot_width=100, plot_height=100)
              bplt.show(self.fig)

          # Update cost function graph
          renderer=self.fig.select(dict(name='top_figure'))
          ds = renderer[0].data_source
          ds.data['y'].append(total[-1])
          ds.data['x'].append(ds.data['x'][-1] + 1)
          bplt.cursession().store_objects(ds)
          #ds.push_notebook()
