import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
import pandas as pd
import sympy as sp
from scipy.optimize import minimize_scalar

#-------- data ~ I.I.D. ---------------------------------------------------------------------------- #
data = pd.read_csv('Advertising.csv')

#__________________ Visualisation des données _____________________________________________#
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

#------ w initial --------------#
'''Apres plusieurs changement de W0 nous avons choisi cette w pour notre modele'''
np.random.seed(21)
w = np.random.randn(2,1)

#----------------------- Le modéle ------------------------------------#
def model(X,w):
    return X.dot(w)
#-------------- fonction de cout ----------------------------------------#
def cost_func(X,y,w):
    return (1/len(y)) * np.sum((model(X,w)-y)**2)
#-------------- grad(f) -------------------------------------------------- #
def grad(X,y,w):
    return (2/len(y)) * X.T.dot(model(X,w)-y)
#------------- fonction pour trouver learning rate optimale --------------- #
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

#----------- La phase d'entrainement ------------------------------------------------ #
def grad_d(X,y,w,ep=0.001):
    learning_rate=alphaSearch(X,y,w)
    while np.linalg.norm(grad(X,y,w)) > ep:
        w = w - learning_rate * grad(X,y,w)
        learning_rate=alphaSearch(X,y,w)
    return w

#--------- Visualisation de résultat et évaluation de modéle ------------- #
w_final=grad_d(X_train,y_train,w)
print(cost_func(X_train,y_train,w_final))
plt.grid()
plt.scatter(X_train[:,0],y_train)
xx = np.linspace(X_train[:,0].min()-1,X_train[:,1].max()+1)
yy = w_final[0]*xx + w_final[1]
plt.xlim(X_train[:,0].min()-1,X_train[:,1].max()+1)
plt.ylim(y_train.min()-1,y_train.max()+1)
plt.title('Visualisation de séparateur')
plt.plot(xx,yy,c='r')
plt.show()
#----------------------------------------------------- Resume -------------------------------------------- #
print('*********************************************** Résumé ********************************************************************')
print('*                                                                                                                          *')
print('*    Nous avons utilisé la métode sample de pandas pour génerer un échatillon aléatoire de 20%                             *')
print(f"*    L équation du modéle est : y={w_final[0]}*x+{w_final[1]}                                                         *")
print(f'*    L erreur de géneralisation : {cost_func(X_test,y_test,w_final)},                                                                      *')
print(f'*    L erreur d appoximation : {cost_func(X_train,y_train,w_final)}                                                                          *')
print(f'*    Les meilleures paramétres sélectionnées pour le modéle sont: ep=0.001, W0 = [-0.05,-0.11] et learning rate adaptatif. *')
print('*                                                                                                                          *')
print('***************************************************************************************************************************')
