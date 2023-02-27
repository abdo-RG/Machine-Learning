import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd
plt.style.use('ggplot')
#_________________________________________________df__________________________________________________________________

df=pd.read_csv('Advertising.csv')

#__________________ Visualisation des données _____________________________________________#
fig = plt.figure()
plt.xlim(df['TV'].min()-5,df['TV'].max()+5)
plt.ylim(df['Sales'].min(),df['Sales'].max()+5)
plt.scatter(df['TV'],df['Sales'])
plt.title('Visualisation des données')
plt.draw()
plt.pause(0.5)
fig.clear()

#---- Division des données entre 80% pour train-set et 20% pour test-set

test_set = df.sample(int(len(df)*0.2),random_state=0) # 20%
train_set = df.drop(test_set.index,axis=0) # 80%
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



#________________________________________________maximisation de la marge_______________________________________________
ep=8
C=10
n=len(X_train)

fun = lambda w: 0.5 * (w[0] ** 2) + C * sum([w[m] for m in range(2, n + 2)]) + C * sum([w[m] for m in range(n + 2, n + n + 2)])
cons = list()
for i, j, k, l in zip(X_train, y_train, range(2, n + 2), range(n + 2, 2 * n + 2)):
    t = {'type': 'ineq', 'fun': lambda w, i=i, j=j, k=k: -j + (i * w[0] + w[1]) + ep + w[k]}
    cons.append(t)
    t={'type': 'ineq', 'fun': lambda w,k=k:     w[k] }
    cons.append(t)
    t = {'type': 'ineq', 'fun': lambda w, i=i, j=j, l=l: j - (i * w[0] + w[1]) + ep + w[l]}
    cons.append(t)
    t = {'type': 'ineq', 'fun': lambda w, l=l: w[l]}
    cons.append(t)

res = minimize(fun, np.random.randn(1,2*n+2), method='SLSQP',jac='2-point',constraints=cons)
theta=res.x

#________________________________________________ploting________________________________________________________________
x1 = np.arange(np.min(X_train),np.max(X_train))
x2 = (theta[0]*x1+theta[1])
x3 = np.arange(np.min(X_train),np.max(X_train))
x4 = (theta[0]*x3+theta[1]-ep)
x5 = np.arange(np.min(X_train),np.max(X_train))
x6 = (theta[0]*x5+theta[1]+ep)
plt.scatter(X_train,y_train)
#plt.axis('scaled')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title('Soft_SVR')
plt.plot(x1, x2,c = 'green')
plt.plot(x3, x4,c = 'red')
plt.plot(x5, x6,c = 'red')
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
print(f"*    L équation du modéle est : y={theta[0]}*x+{theta[1]}                                                  *")
print(f'*    L erreur de géneralisation : {cost_func(X_test,y_test,np.array([theta[0],theta[1]]))},                                                                      *')
print(f'*    L erreur d appoximation : {cost_func(X_train,y_train,np.array([theta[0],theta[1]]))}                                                                           *')
print(f'*    Les meilleures paramétres sélectionnées pour le modéle sont: ep=8, C=10                                               *')
print('*                                                                                                                          *')
print('***************************************************************************************************************************')
