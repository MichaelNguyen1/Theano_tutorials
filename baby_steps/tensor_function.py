import theano
import theano.tensor as T

## Goal is to apply the logistic function element-wise
# Compare two forms of the logistic regression
x = T.dmatrix('x')
s = 1 / (1 + T.exp(-x))
logistic = theano.function([x], s)

s2 = (1 + T.tanh(x / 2)) / 2
logistic2 = theano.function([x], s2)

output = logistic([[0, 1], [-1, -2]])
output2 = logistic2([[0, 1], [-1, -2]])

## Print results
print "Output 1 is: \n", output
print "Output 2 is: \n", output2
print "---------------"


## multiple computations in one step
a, b = T.dmatrices('a', 'b')
diff = a - b
abs_diff = abs(diff)
diff_squared = diff**2

f = theano.function([a,b], [diff, abs_diff, diff_squared])
output3 = f([[1, 1], [1, 1]], [[0, 1], [2, 3]])

## Print results
print " Result of cascaded functions: \n", output3
