import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd

#----------------------------------------DATA I.I.D. ------------------------------------------------- #

data = pd.read_csv('Advertising.csv')

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

y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)



#----------- objectif function --------------------------------------------------------------------#

fct = lambda w : 0.5*(w[0]**2)

#------------------------------------------constraints----------------------------------------------#
cons = ()
ep = 8
for i,j in zip(X_train,y_train):
    cons = cons + ({'type': 'ineq', 'fun': lambda w,i=i,j=j: ep - abs(j - (w[0]*i + w[1])) },)

#--------------------- maximisation de la marge <===> minization using SLSQP méthode ------------------------------------------------#
res = minimize(fct, np.array([1, 1]), method='SLSQP',jac='2-point',constraints=cons)
theta=res.x

#------------------ ploting --------------------------------------------------------------------------#
x1 = np.linspace(X_train.min()-1,X_train.max()+1)
x2 = theta[0]*x1 + theta[1]
x3 = theta[0]*x1 + theta[1] + ep
x4 = theta[0]*x1 + theta[1] - ep
plt.plot(x1,x2,c='r')
plt.plot(x1,x3,'--',c='b')
plt.plot(x1,x4,'--',c='b')
plt.scatter(X_train,y_train)
plt.show()

X_train = np.array(np.hstack((X_train,np.ones(X_train.shape))))
X_test = np.array(np.hstack((X_test,np.ones(X_test.shape))))

#----------------------- Le modéle ------------------------------------#
def model(X,w):
    return X.dot(w)
#-------------- fonction de cout ----------------------------------------#
def cost_func(X,y,w):
    return (1/len(y)) * np.sum((model(X,w)-y)**2)

#----------------------------------------------------- Resume -------------------------------------------- #
print('*********************************************** Résumé ********************************************************************')
print('*                                                                                                                          *')
print('*    Nous avons utilisé la métode sample de pandas pour génerer un échatillon aléatoire de 20%                             *')
print(f"*    L équation du modéle est : y={theta[0]}*x+{theta[1]}                                                         *")
print(f'*    L erreur de géneralisation : {cost_func(X_test,y_test,theta)},                                                                      *')
print(f'*    L erreur d appoximation : {cost_func(X_train,y_train,theta)}                                                                          *')
print(f'*    Les meilleures paramétres sélectionnées pour le modéle sont: ep=8,*')
print('*                                                                                                                          *')
print('***************************************************************************************************************************')
