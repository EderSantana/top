__version__='0.0.1'

import theano
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse
from theano.compat.python2x import OrderedDict

floatX = theano.config.floatX

def AdaGrad(params, cost, lr=1.0, eps=1e-6, lr_rate=None):
    """AdaGrad algorithm proposed was proposed in No More Pesky Learning
    Rates by Schaul et. al.
    
    :param params: list of :class:theano.shared variables to be optimized 
    :param cost: cost function that should be minimized in the optimization
    :param float lr: learning rate
    :param float eps: small constant to avoid division by zero
    :param float lr_rate: learning rate change factor. Ot should be smaller, but close to one
    """
    zero = np.zeros(1).astype(floatX)[0]
    grads = T.grad(cost, params)
    accum = [theano.shared(param.get_value()*zero) for param in params]
    updates = []
    for p, g, a in zip(params, grads, accum):
        a_i = a + g**2
        updates.append((a, a_i))
        updates.append((p, p - lr * g / T.sqrt(a_i + eps)))
    return lr_m_schedule(updates, lr, None, lr_rate, None)

def Adam(params, cost, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
    """Adam algorithm proposed was proposed in Adam: A Method for Stochastic 
    Optimization.
    This code was modified from Newmu's code:
    https://gist.github.com/Newmu/acb738767acb4788bac3
    
    :param params: list of :class:theano.shared variables to be optimized 
    :param cost: cost function that should be minimized in the optimization
    :param float lr: learning rate
    :param float b1: ToDo: WRITEME 
    :param float b2: ToDo: WRITEME
    :param float e: ToDO: WRITEME
    """
    zero = np.zeros(1).astype(floatX)[0]
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(zero)
    i_t = i + 1.
    b1t = 1. - (1. - b1)*e**(-i)
    for p, g in zip(params, grads):
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.) 
        m_t = (b1t * g) + ((1. - b1t) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        m_h = m_t / (1-(1-b1)**i_t)
        v_h = v_t / (1-(1-b2)**i_t)
        p_t = p - lr * m_h / (T.sqrt(v_h) + e) 
        updates.append((m,m_t))
        updates.append((v,v_t))
        updates.append((p,p_t))
    updates.append((i, i_t))
    return updates

def rmsprop(parameters,cost=None,gradients=None,
           updates=None,lr=2e-2, consider_constant = [], epsilon=1e-8,
           momentum = None, lr_rate=None, m_rate=None, 
           g_clip=None, **kwargs):

    rho = .9
    my1 = T.constant(np.array(1.0,dtype=theano.config.floatX))

    if not isinstance(parameters,list):
        parameters = [parameters]
    if gradients == None:
        grads = T.grad(cost,parameters,consider_constant = consider_constant, disconnected_inputs='warn')

    if updates==None:
        updates = []
    for param,grad in zip(parameters,grads):
        if g_clip is not None:
            gnorm = T.sqr(grad).sum()
            grad = ifelse(gnorm>g_clip, g_clip*grad/gnorm, grad)
        scale = my1
        accum  = theano.shared(param.get_value()*0.)
        new_accum = rho * accum + (1 - rho) * grad**2
        updates.append((accum, new_accum))
        if 'scale' in kwargs:
            print 'scaling the lr'
            scale = kwargs['scale']
        if momentum != None:
            mparam = theano.shared(param.get_value()*0.)
            v = momentum * mparam - lr * grad/ T.sqrt(new_accum + epsilon) 
            w = param + momentum * v - lr * grad/ T.sqrt(new_accum + epsilon)
            updates.append((mparam, v))
            updates.append((param, w))
        else:
            updates.append((param, param - scale*lr*grad / T.sqrt(new_accum + epsilon)))

    return lr_m_schedule(updates, lr, momentum, lr_rate, m_rate)

def sgd(parameters,cost=None,gradients=None,
        updates=None,lr=None, consider_constant = [],
        momentum = None, lr_rate=None, m_rate=None, **kwargs):

    my1 = T.constant(np.array(1.0,dtype=theano.config.floatX))
    if not isinstance(parameters,list):
        parameters = [parameters]
    if gradients == None:
        grads = T.grad(cost,parameters,consider_constant = consider_constant,disconnected_inputs='warn')

    if updates==None:
        updates = []
    for param,grad in zip(parameters,grads):
        scale = my1
        if 'scale' in kwargs:
            print 'scaling the lr'
            scale = kwargs['scale']
        if momentum != None:
            mparam = theano.shared(param.get_value()*0.)
            updates.append((param, param - scale * lr * mparam))
            #updates[mparam] = mparam*momentum + (1.-momentum)*grad
            updates.append((mparam, mparam*momentum + grad))
        else:
            updates.append((param, param - scale * lr * grad))

    return lr_m_schedule(updates, lr, momentum, lr_rate, m_rate)

def lr_m_schedule(updates, lr, momentum, lr_rate=None, m_rate=None):
  if lr_rate:
    updates.append((lr, ifelse(lr*lr_rate<1e-5,.1e-5,lr*lr_rate))) #T.minimum(lr * lr_rate, 1e-6)
  if m_rate:
    updates.append((momentum, ifelse(momentum*m_rate>.9,.9,momentum*m_rate))) #T.maximum(momentum * m_rate, .99)
  return updates

class Optimizer():
  """Optim
  """
  def __init__(self, parameters, cost, method='sgd',input=[], givens=None,
               constant=None,learning_rate=.001, momentum=None,
               lr_rate=None, m_rate=None, extra_updates=None,
               g_clip=None):
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
    self.g_clip = g_clip

  def compile(self):
    print "$> Compiling optimizer."

    if self.method.lower() == 'sgd':
      updates = sgd(self.p, cost=self.cost, lr=self.lr, momentum=self.m,
                   lr_rate=self.lr_rate, m_rate=self.m_rate, consider_cosntant=self.cc)
    elif self.method.lower() == 'rmsprop':
      updates = rmsprop(self.p,self.cost, lr=self.lr, momentum=self.m,
                        lr_rate=self.lr_rate, m_rate=self.m_rate, 
                        consider_constant=self.cc, g_clip=self.g_clip)
    elif self.method.lower() == 'adam':
      updates = Adam(self.p, self.cost, lr=self.lr)
    elif self.method.lower() == 'adagrad':
      updates = AdaGrad(self.p, self.cost, lr=self.lr, lr_rate=self.lr_rate)
    else:
      raise NotImplementedError("Optimization method not implemented!")
    
    if self.extra_updates is not None:
        updates.append(self.extra_updates)
    
    updates = OrderedDict(updates)

    if self.input == []:
        self.f = theano.function([], self.cost, updates=updates, givens=self.givens, allow_input_downcast=True)
        self.g = theano.function([], self.cost, givens=self.givens, allow_input_downcast=True)
    else:
      if not isinstance(self.input,list):
          self.input = [self.input]
      self.f = theano.function(self.input, self.cost, updates=updates, givens=self.givens, allow_input_downcast=True)
      self.g = theano.function(self.input, self.cost, givens=self.givens, allow_input_downcast=True)

    return self

  def run(self,niter,*args):
    total = 0.
    if not hasattr(self,'f'):
      self.compile()
    for k in range(niter):
      total += self.f(*args)
    return self, total
  
  '''
  TODO maybe? Add validate_n_save method
  def validate_and_save(valid, path):
      new_cost = valid()
      if new_cost<self.best_cost:
          self.best_cost = new_cost
          pack = {'parameters': self.p, 'iter': iter, 'cost': self.best_cost}
          cPickle.dump(pack,path,-1)
  '''


