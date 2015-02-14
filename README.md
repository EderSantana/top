`top` is a lighweight Theano based optimization module. Right now we have only stochastic gradient descent (SGD) and RMSprop, AdaGrad and Adam but you can do wonders with those.

#Prereqs
Theano, numpy and scipy

#Installation
Just add the present folder to your PYTHONPATH 'export PYTHONPATH=/this/folder:$PYTHONPATH', or 'run pip install' ./ in this directory.

#How to use
Define your function with theano. All the variables you want to optimize should be shared variables. This can be usefull, for instance, to allow you to use the full power of your GPU without needing to move your parameters back and forth to your CPU.

#Quick example
      import theano.tensor as T
      import numpy as np
      from top import Optimizer
      # Define your graph
      b = T.vector()
      A = theano.shared(np.random.randn(3,3).astype(theano.config.floatX))
      x = theano.shared(np.zeros(3, dtype=theano.config.floatX))
      c = theano.shared(np.ones(3, dtype=theano.config.floatX))
      # You finding the solution for a random linear system of equations the adaptive way
      cost = T.sqr(b-T.dot(A,x)+c).sum()
      opt = Optimizer(x, cost, input=b, method='sgd', learning_rate=.1, momentum=.5,
                      lr_rate=.998,m_rate=1.0001)
      bb = np.random.randn(3).astype(theano.config.floatX)
      opt.run(5000,bb)
      print "Expected:"
      print np.dot(np.linalg.inv(A.get_value()),bb+c.get_value())
      print "Found:"
      print x.get_value()
      print 'Try again!' # Random systems may be ill-posed, you know?

