import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from math import *


plt.style.use('ggplot')
#------------------------------------------------------data-------------------------------------------------------------
d,l=datasets.make_circles(n_samples=100,  shuffle=True, noise=0.2, random_state=0, factor=0.3)
l[l==0]=-1
#---------------------------------------data plot before transformation-------------------------------------------------
fig = plt.figure()

plt.title("Transformation Pocket")
plt.scatter(d[:,0], d[:,1], c=l, linewidth=1)
plt.draw()
plt.pause(2)
fig.clear()

#---------------------------------------data plot after transformation-------------------------------------------------
plt.title("Transformation Pocket")
d=np.asarray((d[:,0]**2,d[:,1]**2)).T
plt.axis([np.min(d[:,0])-0.1, np.max(d[:,0])+0.1, np.min(d[:,1])-0.1, np.max(d[:,1])+0.1])
plt.scatter(d[:,0], d[:,1], c=l, linewidth=1)
plt.draw()
plt.pause(1)
fig.clear()
w=np.array([10,-13,18])

#-----------------------------------------------------loss function ----------------------------------------------------
def loss(w,d,l):
    S=0
    m = len(d)
    for i,j in zip(d,l):
        if np.dot(w, np.array([1,i[0], i[1]]))*j < 0:
            S = S + 1
    return S/m

#-----------------------------------------------------plot--------- ----------------------------------------------------
def plt_show(w,d,l,ti):
    x = np.array([min(d[:,0])-1,max(d[:,0])+1])
    y = (-w[1] * x -w[0]) / w[2]
    plt.title("Transformation Pocket")
    plt.axis([np.min(d[:,0])-0.1, np.max(d[:,0])+0.1, np.min(d[:,1])-0.1, np.max(d[:,1])+0.1])
    plt.scatter(d[:,0], d[:,1], c =l, linewidth=1)
    plt.plot(x, y, color="green")
    plt.draw()
    plt.pause(ti)
    fig.clear()
plt_show(w,d,l,0.1)
#-------------------------------------------------Perceptron on transformed data----------------------------------------
ls=loss(w,d,l)
t=0
Tmax=10
for k in range(Tmax):
    for i,j in zip(d,l):
        if np.dot(np.transpose(w), np.array([1,i[0], i[1]])) * j <= 0:
            w=w+j*np.array([1,i[0], i[1]])
            plt_show(w,d,l,0.1)
            t=t+1
    lst=loss(w,d,l)
    if (lst < ls):
        wt = w
        ls = lst
plt_show(w,d,l,2)
#-----------------------------------------------------Final plot--------------------------------------------------------
d,l=datasets.make_circles(n_samples=100,  shuffle=True, noise=0.2, random_state=0, factor=0.3)
plt.title("Transformation Pocket")
x=np.arange(np.min(d[:,0])-1,np.max(d[:,0])+1,0.01)
y=np.arange(np.min(d[:,1])-1,np.max(d[:,1])+1,0.01)
x,y = np.meshgrid(x,y)
z= np.sign(wt[0]+wt[1]*x**2+wt[2]*y**2)
plt.scatter(d[:,0],d[:,1], c = l, linewidth=1)
plt.contourf(x, y, z,1, colors = ['darkblue','yellow'], alpha = .1)
plt.contour(x, y, z, cmap = 'viridis')
plt.show()



