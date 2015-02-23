import numpy as np
import theano
from top import Optimizer
from theano import tensor as T

def test_rmspro():
  '''
  Finds the root of x**
  '''
  x = theano.shared(np.asarray([5.,]).astype(theano.config.floatX))
  cost = T.sqr(x).sum()
  lr_rate = np.asarray(.998).astype(theano.config.floatX)
  learning_rate = np.asarray(.1).astype(theano.config.floatX)
  momentum = np.asarray(.1).astype(theano.config.floatX)
  m_rate = np.asarray(1.0001).astype(theano.config.floatX)
   
  opt = Optimizer(x, cost, method='rmsprop', learning_rate=learning_rate, momentum=momentum,
                  lr_rate=lr_rate,m_rate=m_rate)
  opt.run(5000)
  print "Starting at x=5, after 5000 iterations, we found minimum of x**2 at x = %f" % opt.p[0].get_value()
  assert opt.p[0].get_value()<.00001

def try_adam():
  '''
  Finds the root of x**2 using Adam. This will be a test in the future
  '''
  x = theano.shared(np.asarray([5.,]).astype(theano.config.floatX))
  cost = T.sqr(x).sum()
  learning_rate = np.asarray(.1).astype(theano.config.floatX)
  momentum = np.asarray(.1).astype(theano.config.floatX)
   
  opt = Optimizer(x, cost, method='adam', learning_rate=learning_rate, momentum=momentum)
  opt.run(5000)
  print "Starting at x=5, after 5000 iterations, we found minimum of x**2 at x = %f" % opt.p[0].get_value()
  assert opt.p[0].get_value()<.00001
  
def test_sgd():
  '''
  Solves a random linear function of equation 
  '''
  b = T.vector()
  A = theano.shared(np.random.randn(3,3).astype(theano.config.floatX))
  x = theano.shared(np.zeros(3, dtype=theano.config.floatX))
  c = theano.shared(np.ones(3, dtype=theano.config.floatX))
  cost = T.sqr(b-T.dot(A,x)+c).sum()
  opt = Optimizer(x, cost, input=b, method='sgd', learning_rate=.1, momentum=.5,
                  lr_rate=.998,m_rate=1.0001)
  bb = np.random.randn(3).astype(theano.config.floatX)
  opt.run(5000,bb)
  print "Expected"
  print np.dot(np.linalg.inv(A.get_value()),bb+c.get_value())
  print "Found"
  print x.get_value()
  print 'Try again!'
