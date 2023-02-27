from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import math as mt
import pandas as pd

#------------- data I.I.D. ----------------------------------- #

df = pd.read_csv('sample_perceptron.csv')

#---- Visualisation des données ------------------------------- #

fig = plt.figure()
plt.scatter(df['v1'],df['v2'], c =df['label'])
plt.xlabel("x1")
plt.ylabel("x2")
plt.title('Perceptron')
plt.pause(2)
fig.clear()

#---- Division des données entre 80% pour train-set et 20% pour test-set

test_set = df.sample(int(len(df)*0.2),random_state=0) # 20%
train_set = df.drop(test_set.index,axis=0) # 80%

test_set = np.array(test_set)
train_set = np.array(train_set)

# Nous avons utilisé la métode sample de pandas pour génerer un échatillon aléatoire de 20%

''' Ici l'ensemble des hypothéses est infini, donc on calcule la taille minimale d'échantillon via VC-dimension et la borne de géneralisation
Pour cela on utilise la borne de Blumer et al.,1989 : '''

def gen_bound(ep,delta,vcdim):
    return (1/ep)*(4*mt.log2(2/delta)+8*vcdim*mt.log2(13/ep))

''' Le VC-dimension est 2+1 = 3, Donc Pour ep = 0.5 on aura delta=5.e-4 pour entrainer 80% des données.'''

x = np.array(df)
#------ w initial --------------#
'''Aprés plusieurs changement de w0 nous avons choisi cette w unitial pour notre modele'''
w=[0.9729899,0.40221802,0.27449391]

#-------------------- loss function -------------------------------------------------------------------------------
def loss(w,X):
    s = 0
    for i in range(len(X)):
        if np.sign(np.dot(np.transpose(w),np.array([train_set[i][0],train_set[i][1],1])))!=train_set[i][2]:
            s=s+1
    return s/len(X)

#------------------ La phase d'entrainemen ------------------------------------------------------------------------------------
def Min_loss_func(w0,X):
    ctr = 0
    w = w0
    ls = loss(w,X)
    for i in range(len(X)):
        if train_set[i][2]==-1:
            plt.scatter(train_set[i][0],train_set[i][1],color='red')
        if train_set[i][2]==1:
            plt.scatter(train_set[i][0],train_set[i][1],color='blue')
    t = np.arange(train_set[:,0].min()-1,train_set[:,0].max()+1,0.1)
    y = (-w[0]/w[1])*t - w[0]/w[2]
    plt.axis([train_set[:,0].min()-1,train_set[:,0].max()+1,train_set[:,1].min()-1,train_set[:,1].max()+1])
    plt.plot(t,y,color='green')
    plt.title('data linearly separable')
    plt.draw()
    plt.pause(2)
    fig.clear()
    while ls!=0:
        for i in range(len(X)):
            if np.sign(np.dot(w,np.array([train_set[i][0],train_set[i][1],1])))!=train_set[i][2]:
                w = w + train_set[i][2]*(np.array([train_set[i][0],train_set[i][1],1]))
                ctr = ctr+1
                for j in range(len(X)):
                    if train_set[j][2]==-1:
                        plt.scatter(train_set[j][0],train_set[j][1],color='red')
                        if i==j:
                            plt.plot(train_set[j][0],train_set[j][1],'ro',markersize = 15,alpha=.5)
                    if train_set[j][2]==1:
                        plt.scatter(train_set[j][0],train_set[j][1],color='blue')
                        if i==j:
                            plt.plot(train_set[j][0],train_set[j][1],'bo',markersize = 15,alpha=.5)
                t = np.arange(train_set[:,0].min()-1,train_set[:,0].max()+1,0.1)
                y = (-w[0]/w[1])*t - w[2]/w[1]
                plt.axis([train_set[:,0].min()-1,train_set[:,0].max()+1,train_set[:,1].min()-1,train_set[:,1].max()+1])
                plt.plot(t,y,color='green')
                plt.title('Acadimic perceptron (data linearly separable)')
                plt.draw()
                plt.pause(0.5)
                fig.clear()
        ls=loss(w,X)

    return w,ls,ctr


#----------------------------- ploting ----------------------------------#

w,ls,ctr = Min_loss_func(w,train_set)

for i in range(len(train_set)):
    if train_set[i][2]==-1:
        plt.scatter(train_set[i][0],train_set[i][1],color='red')
    if train_set[i][2]==1:
        plt.scatter(train_set[i][0],train_set[i][1],color='blue')

t = np.arange(train_set[:,0].min()-1,train_set[:,0].max()+1,0.1)
y = (-w[0]/w[1])*t - (w[2]/w[1])
plt.axis([train_set[:,0].min()-1,train_set[:,0].max()+1,train_set[:,1].min()-1,train_set[:,1].max()+1])
plt.plot(t,y,color='k')
plt.title("Best separator")
plt.show()

#----------------------------------------------------- Resume -------------------------------------------- #
print('*********************************************** Résumé ********************************************************************')
print('*                                                                                                                          *')
print('*    Les meilleures paramétres sélectionnées pour le modéle sont : W0 = [1,0.5,0.5], splting data avec random state=0      *')
print('*    Nous avons utilisé la méthode sample de pandas pour génerer un échatillon aléatoire de 20%                             *')
print('*    Le VC-dimension est 2+1 = 3, Donc Pour ep = 0.5 on aura delta=5.e-4 pour entrainer 80% des données.                   *')
print(f"*    L équation du modéle est : y={-w[0]/w[1]}*x+{- (w[2]/w[1])}                                                  *")
print(f'*    L erreur de géneralisation : {loss(w,test_set)}                                                                                      *')
print(f'*    L erreur d appoximation : {loss(w,train_set)}                                                                                         *')
print('*                                                                                                                          *')
print('***************************************************************************************************************************')