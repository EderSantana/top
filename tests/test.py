import numpy as np
import theano
from top import Optimizer

def test1():
  '''
  Finds the root of x**
  '''
  x = theano.shared(np.asarray(5.).astype(theano.config.floatX))
  cost = T.sqr(x)
  opt = Optimizer(x, cost, method='rmsprop', learning_rate=.1, momentum=.5,
                  lr_rate=.998,m_rate=1.0001)
  opt.run(5000)
  print "Starting at x=5, after 5000 iterations, we found minimum of x**2 at x = %f" % opt.p.get_value()

def test2():
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
