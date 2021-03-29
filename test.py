import numpy as np

a = np.arange(20).reshape(10,2)
print(a[(a[:,0]>15)&(a[:,1]>15)])