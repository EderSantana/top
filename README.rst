`top` is a lighweight Theano based optimization module. Right now we
have only stochastic gradient descent (SGD) and RMSprop, AdaGrad and
Adam but you can do wonders with those.

Prereqs
=======

Theano, numpy and scipy

Installation
============

Just add the present folder to your PYTHONPATH ‘export
PYTHONPATH=/this/folder:$PYTHONPATH’

How to use
==========

Define your function with theano. All the variables you want to optimize
should be shared variables. This can be usefull, for instance, to allow
you to use the full power of your GPU without needing to move your
parameters back and forth to your CPU.
