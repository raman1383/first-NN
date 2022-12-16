import math

"""Exponentiation -> to remove negatives without losing meaning"""
layer_outputs = [4.8, 1.21, 2.385]
E = math.e
exp_values = []
for output in layer_outputs:
    exp_values.append(E**output)
print(exp_values)


"""normalization -> to avoid big numbers"""

norm_base = sum(exp_values)
norm_values = []
for values in exp_values:
    norm_values.append(values/norm_base)
print(norm_values)
print(sum(norm_values))
