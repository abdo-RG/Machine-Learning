import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sp
from scipy.optimize import minimize_scalar

#-------- data ~ I.I.D. ---------------------------------------------------------------------------- #
data = pd.read_csv("Advertising.csv")

#__________________ Visualisation des données _____________________________________________#

fig = plt.figure()
plt.xlim(data['TV'].min()-5,data['TV'].max()+5)
plt.ylim(data['Sales'].min(),data['Sales'].max()+5)
plt.scatter(data['TV'],data['Sales'])
plt.title('Visualisation des données')
plt.draw()
plt.pause(0.5)
fig.clear()

#---- Division des données entre 80% pour train-set et 20% pour test-set

test_set = data.sample(int(len(data)*0.2),random_state=0) # 20%
train_set = data.drop(test_set.index,axis=0) # 80%
# Nous avons utilisé la métode sample de pandas pour génerer un échatillon aléatoire de 20%

# Prétraitement des données
y_test = np.array(test_set['Sales'])
y_train = np.array(train_set['Sales'])

X_train = np.array(train_set['TV'])
X_test = np.array(test_set['TV'])

X_train = X_train.reshape(X_train.shape[0],1)
X_test = X_test.reshape(X_test.shape[0],1)

X_train = np.array(np.hstack((X_train,np.ones(X_train.shape))))
X_test = np.array(np.hstack((X_test,np.ones(X_test.shape))))

y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)
# mise à l'échelle des données pour eviter l'overfiting et augmenter le performance
X_train[:,0] = (X_train[:,0]-X_train[:,0].mean())/X_train[:,0].std()
X_test[:,0] = (X_test[:,0]-X_test[:,0].mean())/X_test[:,0].std()
y_test = (y_test-y_test.mean())/y_test.std()
y_train = (y_train-y_train.mean())/y_train.std()

#__________Input features___________________________________________________________________________ #
x = np.array(data["Sales"])
x = x.reshape(x.shape[0],1)
X = np.array(np.hstack((x,np.ones(x.shape))))
#___________labels__________________________________________________________________________________ #

Y = np.array(data["TV"])
Y = Y.reshape(Y.shape[0],1)

#6- Now, fit the data with polynomial regression model. Try different polynomial orders 
# (𝑄 = 2, 3, 4), and in each time compute the empirical error Ls(ℎpoly). 
# Pour Q = 2
X_Q2 = np.array(np.hstack((x**2,X)),dtype=float)
X_Q3 = np.array(np.hstack((x**3,X_Q2)),dtype=float)
X_Q4 = np.array(np.hstack((x**4,X_Q3)),dtype=float)

#Mise à l'échelle des données

for i in range(X_Q2.shape[1]-1):
    X_Q2[:,i]=(X_Q2[:,i]-X_Q2[:,i].mean()/X_Q2[:,i].std())

for i in range(X_Q3.shape[1]-1):
    X_Q3[:,i]=(X_Q3[:,i]-X_Q3[:,i].mean()/X_Q3[:,i].std())

for i in range(X_Q4.shape[1]-1):
    X_Q4[:,i]=(X_Q4[:,i]-X_Q4[:,i].mean()/X_Q4[:,i].std())

Y = (Y-Y.mean())/Y.std()

np.random.seed(9)
w_2 = np.random.randn(3,1)
w_3 = np.random.randn(4,1)
w_4 = np.random.randn(5,1)
# w_3 = np.zeros((4,1))

def model(X,w):
    return X.dot(w)

def cost_func(X,y,w):
    return (1/len(y)) * np.sum((model(X,w)-y)**2)
# print(cost_func(X_Q4,Y,w_4))
def grad(X,y,w):
    return (2/len(y)) * X.T.dot(model(X,w)-y)

def alphaSearch(X,y,w):
  grd = grad(X,y,w)
  a = sp.Symbol("a")
  wk = w - a*grd
  f = str(cost_func(X,y,wk))
  def fun(a):
    fu = eval(f)
    return fu
  alpha = minimize_scalar(fun)
  return alpha.x

cost_history=np.zeros(1000)
def gradient_descent(X,y,w,ep=0.3):
    global j
    j = 0
    learning_rate=alphaSearch(X,y,w)
    while np.linalg.norm(grad(X,y,w)) > ep:
        w = w - learning_rate * grad(X,y,w)
        learning_rate=alphaSearch(X,y,w)
        cost_history[j]=cost_func(X,y,w)
        j+=1
        print(cost_history[j])
    return w,cost_history
w_Q2_f,cost_history_2=gradient_descent(X_Q2,Y,w_2)
print(cost_history_2[j])
w_Q3_f,cost_history_3=gradient_descent(X_Q3,Y,w_3)
print(cost_history_3[j])
w_Q4_f,cost_history_4=gradient_descent(X_Q4,Y,w_4)
print(cost_history_3[j])
plt.plot(x,model(X_Q2,w_Q2_f),c='r')
plt.scatter(x,Y)
plt.show()

#the best model is : model(X_Q2,w_Q2_f) #
#the best model between linear regression model and the best polynomial regression model is this last one #

# w_0_1 ,ch= gradient_descent(X_Q2,Y,w_2,ep=0.1)
# w_0_2 ,ch= gradient_descent(X,Y,w_2,ep=0.2)
# w_0_3 ,ch= gradient_descent(X,Y,w_2,ep=0.3)
# w_0_4 ,ch= gradient_descent(X,Y,w_2,ep=0.4)
# plt.plot(x,model(X_Q2,w_0_1),c='r')
# plt.scatter(x,Y)
# plt.show()


'''Comments : the best model for this data is the model with Q=2, when we choose learning rate constant, the algorithm may diverge,
so the best solution is with adaptave learning rate'''