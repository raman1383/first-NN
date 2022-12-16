"""Exponentiation + Normalization = Softmax"""

import math

import numpy

"""Exponentiation -> to remove negatives without losing meaning"""
layer_outputs = [4.8, 1.21, 2.385]
E = math.e
exp_values = numpy.exp(layer_outputs)


"""Normalization -> to avoid big numbers"""
norm_values = exp_values / numpy.sum(exp_values)


print(norm_values)
print(sum(norm_values))
