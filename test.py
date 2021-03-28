import numpy as np

a = np.zeros((5,5))
b = np.ones((3,3))
a[-3:,-3:] += b
print(a)