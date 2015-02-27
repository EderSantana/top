__version__='0.0.1'

import theano
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse
from theano.compat.python2x import OrderedDict
from top.optimizer import Optimizer
from top import update_rules as up

'''
  TODO maybe? Add validate_n_save method
  def validate_and_save(valid, path):
      new_cost = valid()
      if new_cost<self.best_cost:
          self.best_cost = new_cost
          pack = {'parameters': self.p, 'iter': iter, 'cost': self.best_cost}
          cPickle.dump(pack,path,-1)
'''
