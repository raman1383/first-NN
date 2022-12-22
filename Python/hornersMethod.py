# implementation of Horner Method
# for Polynomial Evaluation

# returns value of poly[0]x(n-1)
# + poly[1]x(n-2) + .. + poly[n-1]
def horner(poly, n, x):

    # Initialize result
    result = poly[0]

    # Evaluate value of polynomial
    # using Horner's method
    for i in range(1, n):

        result = result*x + poly[i]

    return result

# Driver program to
# test above function.


# Let us evaluate value of
# 2x3 - 6x2 + 2x - 1 for x = 3
poly = [2, -6, 2, -1]
x = 3
n = len(poly)

print("Value of polynomial is ", horner(poly, n, x))
