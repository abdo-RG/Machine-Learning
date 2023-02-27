import pandas as pd
import numpy as np
import sympy as sp 
import matplotlib.pyplot as plt
from numpy import linalg as la
from scipy.optimize import minimize_scalar
import random as rand
fig = plt.figure()
df=pd.read_csv('dataset-cars.csv')
x=np.array(df['speed'])
y=np.array(df['dist'])
plt.scatter(x,y)
plt.pause(1)
fig.clear()
m=len(x)
well = list()
for i in range(m):
    well.append(np.array([1,x[i]]))
dataset = np.array(well)
def loss(weights,data,label):
    return 0.5*(np.dot(weights,data)-label)**2
def grad_loss(weights,data,label):
    return np.dot((np.dot(weights,data)-label),data)
def alphaSearch(w,data,label):
  grad = grad_loss(w,data,label)
  a = sp.Symbol("a")
  wk = w - a*grad
  f = str(loss(wk,data,label))
  def fun(a):
    fu = eval(f)
    return fu
  alpha = minimize_scalar(fun)
  return alpha.x
w=np.array([1,-5])
index=rand.randint(0,m-1)
data=dataset[index]
label=y[index]
gls=grad_loss(w,data,label)
ls=loss(w,data,label)
while la.norm(gls) > 0.01:
    index=rand.randint(0,m-1)
    data=dataset[index]
    label=y[index]
    alpha=alphaSearch(w,data,label)
    print(alpha)
    w=w-alpha*gls
    a = np.array([np.min(x)-5,np.max(x)+5])
    b = (w[1] * a +w[0])
    plt.plot(a, b, color="green")
    plt.scatter(x,y)
    plt.axis([np.min(x)-5, np.max(x)+5, np.min(y)-5, np.max(y)+5])
    plt.draw()
    plt.pause(0.1)
    fig.clear()
    gls=grad_loss(w,data,label)
    ls=loss(w,data,label)
a = np.array([np.min(x)-5,np.max(x)+5])
b = (w[1] * a +w[0])
plt.plot(a, b, color="green")
plt.scatter(x,y)
plt.axis([np.min(x)-5, np.max(x)+5, np.min(y)-5, np.max(y)+5])
plt.show()
