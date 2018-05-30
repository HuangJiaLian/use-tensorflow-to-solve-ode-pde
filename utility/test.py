
# coding: utf-8

# In[49]:


import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 

NT = 401
NX = 33

# Load text from txt file 
data = np.loadtxt('solSinEF.txt')
# print(data)

# 切片找到t,x,u
t = data[:,0]
t = t[:NT]
# print(t, t.shape)

x = np.linspace(0,1,NX)
# x = x[:33]
# print(x, x.shape)

u = data[:,2]
print(u, u.shape)

fig = plt.figure()  
ax = Axes3D(fig)  
ax.set_xlabel('T')
ax.set_ylabel('X')
ax.set_zlabel('U')  

T, X = np.meshgrid(t,x)
# print(X, X.shape)
# print(T, T.shape)
U = u.reshape(NX,NT,order='C')
# U = u.reshape(401,33,order='F')
surfaces = ax.plot_surface(T,X,U,rstride=1,cstride=1,cmap=plt.cm.jet)
print(T)
print(X)
print(U)
plt.pause(1)
plt.show()

