
# coding: utf-8

# In[50]:


import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 

NT = 33
NX = 33

FILENAME = 'dat32x32.txt'
# Load text from txt file 
data = np.loadtxt(FILENAME)
# print(data)

# 切片找到t,x,u
t = data[:,0]
t = t[:NT]
# print(t, t.shape)

x = np.linspace(0,1,NX)
# print(x, x.shape)

u = data[:,2]
# print(u, u.shape)

fig = plt.figure()  
ax = Axes3D(fig)  
ax.set_xlabel('T')
ax.set_ylabel('X')
ax.set_zlabel('U')  

T, X = np.meshgrid(t,x)
# print(X, X.shape)
# print(T, T.shape)
U = u.reshape(NX,NT,order='C')
surfaces = ax.plot_surface(T,X,U,rstride=1,cstride=1,cmap=plt.cm.jet)
print(T)
print(X)
print(U)
plt.pause(1)
plt.show()

