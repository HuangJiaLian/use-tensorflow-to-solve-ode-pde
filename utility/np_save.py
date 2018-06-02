import numpy as np 

a = np.array([[1,2,3],[4,5,6]])
np.savetxt('test1.txt', a, fmt='%f')
b = np.loadtxt('test1.txt', dtype=np.float32)
print(b)