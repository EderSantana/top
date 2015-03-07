import theano
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse

floatX = theano.config.floatX
def adam(params, cost, lr=0.0002, b1=0.1, b2=0.001, e=1e-8, grad_clip=None):
    """Adam algorithm proposed was proposed in Adam: A Method for Stochastic
    Optimization.
    This code was modified from Newmu's (Alec Radford) code:
    https://gist.github.com/Newmu/acb738767acb4788bac3

    :param params: list of :class:theano.shared variables to be optimized
    :param cost: cost function that should be minimized in the optimization
    :param float lr: learning rate
    :param float b1: ToDo: WRITEME
    :param float b2: ToDo: WRITEME
    :param float e: ToDO: WRITEME
    """
    updates = []
    grads = T.grad(cost, params)
    zero = np.zeros(1).astype(floatX)[0]
    i = theano.shared(zero)
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        if grad_clip is not None:
            gnorm = T.sqrt(T.sqr(g).sum())
            ggrad = T.switch(T.ge(gnorm,grad_clip),
                             grad_clip*g/gnorm, g)
        else:
            ggrad = g
        m = theano.shared(p.get_value() * 0.)
        v = theano.shared(p.get_value() * 0.)
        m_t = (b1 * ggrad) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(ggrad)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates

def adagrad(params, cost, lr=1.0, eps=1e-6, lr_rate=None):
    """AdaGrad algorithm proposed was proposed in No More Pesky Learning
    Rates by Schaul et. al.

    :param params: list of :class:theano.shared variables to be optimized
    :param cost: cost function that should be minimized in the optimization
    :param float lr: learning rate
    :param float eps: small constant to avoid division by zero
    :param float lr_rate: learning rate change factor. Ot should be smaller,
     but close to one
    """
    zero = np.zeros(1).astype(floatX)[0]
    grads = T.grad(cost, params)
    accum = [theano.shared(param.get_value()*zero) for param in params]
    updates = []
    for p, g, a in zip(params, grads, accum):
        a_i = a + g**2
        updates.append((a, a_i))
        updates.append((p, p - lr * g / T.sqrt(a_i + eps)))
    return _lr_m_schedule(updates, lr, None, lr_rate, None)

def Adam2(params, cost, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
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
           grad_clip=None, **kwargs):

    rho = .9
    my1 = T.constant(np.array(1.0,dtype=theano.config.floatX))

    if not isinstance(parameters,list):
        parameters = [parameters]
    if gradients == None:
        grads = T.grad(cost,parameters,consider_constant = consider_constant,
                       disconnected_inputs='warn')

    if updates==None:
        updates = []
    for param,grad in zip(parameters,grads):
        if grad_clip is not None:
            gnorm = T.sqrt(T.sqr(grad).sum())
            ggrad = T.switch(T.ge(gnorm,grad_clip),
                             grad_clip*grad/gnorm, grad)
        else:
            ggrad = grad
        scale = my1
        accum  = theano.shared(param.get_value()*0.)
        new_accum = rho * accum + (1 - rho) * ggrad**2
        updates.append((accum, new_accum))
        if 'scale' in kwargs:
            print 'scaling the lr'
            scale = kwargs['scale']
        if momentum != None:
            mparam = theano.shared(param.get_value()*0.)
            v = momentum * mparam - lr * ggrad/ T.sqrt(new_accum + epsilon)
            w = param + momentum * v - lr * ggrad/ T.sqrt(new_accum + epsilon)
            updates.append((mparam, v))
            updates.append((param, w))
        else:
            updates.append((param, param - scale*lr*ggrad / T.sqrt(new_accum + epsilon)))

    return _lr_m_schedule(updates, lr, momentum, lr_rate, m_rate)

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

    return _lr_m_schedule(updates, lr, momentum, lr_rate, m_rate)

def _lr_m_schedule(updates, lr, momentum, lr_rate=None, m_rate=None):
  if lr_rate:
    updates.append((lr, ifelse(lr*lr_rate<1e-5,.1e-5,lr*lr_rate))) #T.minimum(lr * lr_rate, 1e-6)
  if m_rate:
    updates.append((momentum, ifelse(momentum*m_rate>.9,.9,momentum*m_rate))) #T.maximum(momentum * m_rate, .99)
  return updates
