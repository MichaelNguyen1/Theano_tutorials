import numpy
import theano.tensor as T
from theano import function

## Set up necessary variables/functions
# the variables x,y are instances of class theano.tensor.var.TensorVariable
# the variables x,y are assigned the theano type dscalar in the type field
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y

# function( [ input variables ], function )
f = function([x,y],z)

## Print out test cases
# assign x's and y's values as 2 and 3 respectively:
print("Result of 2 + 3: %d" % f(2,3))

# compare f(16.3, 12.1) to 28.4
output = numpy.allclose(f(16.3, 12.1), 28.4)
print("Is 16.3 + 12.1 = 28.4: %s" % output)
