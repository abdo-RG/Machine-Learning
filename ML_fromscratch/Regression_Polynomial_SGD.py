import pandas as pd
import numpy as np
import sympy as sp 
import matplotlib.pyplot as plt
from numpy import linalg as la
from scipy.optimize import minimize_scalar
import random as rand
fig = plt.figure()
df=pd.read_csv('dataset-14550.csv')
x=np.array(df['temperature'])
y=np.array(df['pressure'])
plt.scatter(x, y)
plt.pause(2)
fig.clear()
z_1=x
z_2=x**2
print(y)
well = list()
for i in range(len(z_1)):
    well.append(np.array([z_1[i],z_2[i]]))
dataset = np.array(well)
dataset=np.insert(dataset,0,1,axis=1)
m=len(z_1)
print(m)
#__________________________________________________________________________________________________________________________________
def loss(weights,data,label):
    return 0.5*(np.dot(weights,data)-label)**2
def grad_loss(weights,data,label):
    return np.dot((np.dot(weights,data)-label),data)
#__________________________________________________________________________________________________________________________________
def alphaSearch(w):
  grad = grad_loss(w,data,label)
  a = sp.Symbol("a")
  wk = w - a*grad
  f = str(loss(wk,data,label))
  def fun(a):
    fu = eval(f)
    return fu
  alpha = minimize_scalar(fun)
  return alpha.x
#__________________________________________________________________________________________________________________________________
W=np.array([[1,0,1],[1,1,1]])
for w in W:
    print("______________________________")
    index=rand.randint(0,m-1)
    data=dataset[index]
    label=y[index]
    gls=grad_loss(w,data,label)
    glsb=gls
    ls=loss(w,data,label)
    t=0
    while la.norm(gls) > 0.001 and t <200:
        glst=gls
        for i in range(3):
            if w[i] ==0:
                glst[i]=0
        alpha=alphaSearch(w)
        w=w-alpha*glst
        plt.scatter(x, y)
        a = np.arange(min(x),max(x))
        b = w[0] + w[1]*a + w[2]*a**2
        plt.plot(a, b, color="green")
        plt.draw()
        plt.pause(0.01)
        fig.clear()
        gls=grad_loss(w,data,label)
        ls=loss(w,data,label)
        if la.norm(gls)<la.norm(glsb):
            glsb=gls
            wb=w
        t=t+1
        print(la.norm(gls))
plt.scatter(x, y)
a = np.arange(min(x),max(x))
b = wb[0] + wb[1]*a + wb[2]*a**2
plt.plot(a, b, color="green")
print("the loss function is ",ls)
print("the gradloss is ",glsb)
plt.show()
