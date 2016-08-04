import numpy
import theano
import theano.tensor as T
from theano import pp

x = T.dscalar('x')
y = x ** 2
gy = T.grad(y,x)
print(pp(gy))

f = theano.function([x], gy)
print(f(4))
print(numpy.allclose(f(94.2), 188.4))
