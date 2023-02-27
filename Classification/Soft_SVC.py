import numpy as np
import math as mt
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd

#----------------------------------------DATA I.I.D. ------------------------------------------------- #

data = pd.read_csv('soft_svc.csv')

#---------------------- Visualisation des données ---------------------------------------------------- #

fig = plt.figure()
plt.scatter(data['feature1'],data['feature2'], c =data['label'])
plt.axis('scaled')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title('Hard_SVM')
plt.pause(2)
fig.clear()

#---- Division des données entre 80% pour train-set et 20% pour test-set

test_set = data.sample(int(len(data)*0.2),random_state=0) # 20%
train_set = data.drop(test_set.index,axis=0) # 80%

y_test = np.array(test_set['label'])
y_train = np.array(train_set['label'])

test_set.drop(['label'],axis=1,inplace=True)
train_set.drop(['label'],axis=1,inplace=True)

test_set = np.array(test_set)
train_set = np.array(train_set)



# Nous avons utilisé la métode sample de pandas pour génerer un échatillon aléatoire de 20%

''' Ici l'ensemble des hypothéses est infini, donc on calcule la taille minimale d'échantillon via VC-dimension et la borne de géneralisation
Pour cela on utilise la borne de Blumer et al.,1989 : '''

def gen_bound(ep,delta,vcdim):
    return (1/ep)*(4*mt.log2(2/delta)+8*vcdim*mt.log2(13/ep))

''' Le VC-dimension est 2+1 = 3, Donc Pour ep = 0.5 on aura delta=5.e-4 pour entrainer 80% des données.'''



#________________________________________________Constarints____________________________________________________________
C=10
n = len(train_set)
fun = lambda w: 0.5*(w[0]**2+w[1]**2)+C*sum([ w[m] for m in range(3,n+3)])
cons=list()
for i,j,k in zip(train_set,y_train,range(3,n+3)):
    f= {'type': 'ineq', 'fun': lambda w,i=i,j=j,k=k:    i[0]*j *w[0]+ i[1]*j *w[1]+ j *w[2] -1+w[k]}
    cons.append(f)
for i in range(3,n+3):
    f={'type': 'ineq', 'fun': lambda w,i=i:     w[i] }
    cons.append(f)
cons = tuple(cons)

res = minimize(fun, np.random.randn(1,n+3), method='SLSQP',jac='2-point',constraints=cons)
theta=res.x
#__________________________________________________Ploting______________________________________________________________
x1 = np.arange(-1,8)
x2 = (-theta[1]*x1-theta[2])/theta[0]
x3 = np.arange(-1,8)
x4 = (-theta[1]*x3-theta[2]-1)/theta[0]
x5 = np.arange(-1,8)
x6 = (-theta[1]*x5-theta[2]+1)/theta[0]
plt.scatter(train_set[:,0],train_set[:,1], c = y_train)
plt.axis('scaled')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title('S_SVC')
plt.plot(x1, x2,c = 'green')
plt.plot(x3, x4,c = 'red')
plt.plot(x5, x6,c = 'red')
plt.show()

#-------------------- loss function -------------------------------------------------------------------------------
def loss(w,X,y):
    s = 0
    for i in range(len(X)):
        if np.sign(np.dot(w,np.array([X[i][0],X[i][1],1])))!=y[i]:
            s=s+1
    return s/len(X)

#----------------------------------------------------- Resume -------------------------------------------- #
print('*********************************************** Résumé ********************************************************************')
print('*                                                                                                                          *')
print('*    Nous avons utilisé la métode sample de pandas pour génerer un échatillon aléatoire de 20%                             *')
print('*    Le VC-dimension est 2+1 = 3, Donc Pour ep = 0.5 on aura delta=0.4 pour entrainer 80% des données.                     *')
print(f"*    L équation du modéle est : y={-theta[0]/theta[1]}*x+{- (theta[2]/theta[1])}                                               *")
print(f'*    L erreur de géneralisation : {loss(np.array([theta[0],theta[1],theta[2]]),test_set,y_test)}, cette resultat est attendue car test_set est presque sans bruit.                   *')
print(f'*    L erreur d appoximation : {loss(np.array([theta[0],theta[1],theta[2]]),train_set,y_train)}                                                                                      *')
print('*                                                                                                                          *')
print('***************************************************************************************************************************')