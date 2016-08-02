import numpy
import theano.tensor as T
from theano import function

## Set up necessary variables/functions
# the variables x,y are instances of class theano.tensor.var.TensorVariable
# the variables x,y are assigned the theano type dscalar in the type field
# more specifically, dscalar handles scalars of double types
x = T.dscalar('x')
y = T.dscalar('y')
z = x + y

# function( [ input variables ], function )
f = function([x,y],z)

## Set up same problem with matrices
xm = T.dmatrix('xm')
ym = T.dmatrix('ym')
zm = xm + ym
fm = function([xm, ym], zm)


## Print out test cases
# assign x's and y's values as 2 and 3 respectively:
print("Result of 2 + 3: %d" % f(2,3))

# compare f(16.3, 12.1) to 28.4
output = numpy.allclose(f(16.3, 12.1), 28.4)
print("Is 16.3 + 12.1 = 28.4: %s" % output)

# add [[1, 2], [3, 4]] to [[10, 20], [30, 40]]
outputm = fm( numpy.array([[1,2], [3,4]]), numpy.array([[10, 20], [30, 40]]) )
print "Adding two matrices: \n", outputm
