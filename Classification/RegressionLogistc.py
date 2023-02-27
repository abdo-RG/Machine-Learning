import numpy as np
import math as mt
import matplotlib.pyplot as plt
import pandas as pd
import mpl_toolkits.mplot3d

fig = plt.figure()

#------------ l'importation des données -------------------------------- #
data_bnr = pd.read_csv('binary.csv')
data_bnr.drop('rank',axis=1,inplace=True)


#---- Division des données entre 80% pour train-set et 20% pour test-set

test_set = data_bnr.sample(int(len(data_bnr)*0.2),random_state=0) # 20%
train_set = data_bnr.drop(test_set.index,axis=0) # 80%
# Nous avons utilisé la métode sample de pandas pour génerer un échatillon aléatoire de 20%

''' Ici l'ensemble des hypothéses est infini, donc on calcule la taille minimale d'échantillon via VC-dimension et la borne de géneralisation
Pour cela on utilise la borne de Blumer et al.,1989 : '''

def gen_bound(ep,delta,vcdim):
    return (1/ep)*(4*mt.log2(2/delta)+8*vcdim*mt.log2(13/ep))

''' Le VC-dimension est 2+1 = 3, Donc Pour ep = 0.5 on aura delta=5.e-4 pour entrainer 80% des données.'''

# Prétraitement des donnéees

y_testt = np.array(test_set['admit'])
y_traint = np.array(train_set['admit'])

test_set.drop(['admit'],axis=1,inplace=True)
train_set.drop(['admit'],axis=1,inplace=True)
#--------- colonne des biais ------------------#
test_set.insert(0,'admit',1,allow_duplicates=False)
train_set.insert(0,'admit',1,allow_duplicates=False)
# mise à l'échelle des données pour eviter l'overfiting et augmenter le performance

test_sett = np.array(test_set)
train_sett = np.array(train_set)

train_sett[:,1] = (train_sett[:,1]-train_sett[:,1].mean())/train_sett[:,1].std()
test_sett[:,1] = (test_sett[:,1]-test_sett[:,1].mean())/test_sett[:,1].std()

#------ w initial --------------#
'''Apres plusieurs changement de W0 nous avons choisi cette w pour notre modele'''
w=[1.01995544,-0.567118,1.13679672]

#---- fonction sigmoid -------------- #
def sigmoid(x, weight):
    z = x.dot(weight)
    return 1 / (1 + np.exp(-z))
#----- loss function -------------- #
def lossfunction(x, y, w):
    m = len(x)
    som =0
    for i in range(m):
         som += np.log(1 + np.exp(-y[i]*(np.dot(w, x[i]))))
    return som/m

# Pour notre dataset les données ne sont pas trop large, donc on n'est pas obliger de faire le gradient stochastique

def gradient(x, y, w):
    som = 0
    m = len(x)
    for i in range(m):
        t1 = (-y[i] * np.exp(-y[i] * np.dot(w, x[i])))
        t2 = 1/(1 + np.exp(-y[i] * np.dot(w, x[i])))
        som +=  (t1/t2) * x[i]
    return som/m

# La phase d'entrainement 
def regressionLogistic(x, y, w, learning_rate= 1, delta = 0.005):
    cptr = 0
    gradloss = gradient(x,y, w)
    while np.linalg.norm(gradloss) > delta :
        w = w - learning_rate * gradloss
        gradloss = gradient(x, y, w)
        plt.xlim(-3.5,2.5)
        plt.ylim(2,4.4)
        plt.scatter(x[:, 1], x[:, 2], s=40, c=y,cmap=plt.cm.Spectral)
        t = np.arange(-4,4,0.01)
        z = (-w[0]/w[1])*t - w[2]/w[1]
        plt.plot(t, z, color='green')
        plt.draw()
        plt.pause(0.5)
        fig.clear()
    loss = lossfunction(x, y, w)
    return w, loss, cptr

theta, loss, compteur = regressionLogistic(train_sett, y_traint, w)

# fonction de decision
def decision_fun(w,set):
    l =  []
    l0 = 0
    l1 = 0
    for i in range(len(set)):
    # print(sigmoid(test_sett[i,:],theta))
        if sigmoid(set[i,:],w) < 0.9995:
            l.append(0)
            l0+=1
        else:
            l.append(1)
            l1+=1
    return l

# Evaluation du modèle
def accuracy(y_vraie,y_pred):
    mc=0
    for j in range(len(y_vraie)):
        if y_vraie[j]!=y_pred[j]:
            mc+=1
    return mc/len(y_vraie)


plt.title('Séparateur finale')
plt.xlim(-3.5,2.5)
plt.ylim(2,4.4)
plt.scatter(train_sett[:, 1], train_sett[:, 2], s=40, c = y_traint, cmap=plt.cm.Spectral)
if ( theta[1]!= 0 ):
    t = np.linspace(-4, 4, 2)
    z = (-theta[0]/theta[1])*t - theta[2]/theta[1]

plt.plot(t, z, color='black')
plt.show()

#----------------------------------------------------- Resume -------------------------------------------- #
print('*********************************************** Résumé ******************************************************************************************')
print('*                                                                                                                                                ')
print('*    Les meilleures paramétres sélectionnées pour le modéle sont : learning_rate= 1, delta = 0.005, W0 = [1.012,-0.567,1.136] et le seuil 0.9995*')
print('*    Nous avons utilisé la méthode sample de pandas pour génerer un échatillon aléatoire de 20%                                                 *')
print('*    Le VC-dimension est 2+1 = 3, Donc Pour ep = 0.5 on aura delta=5.e-4 pour entrainer 80% des données.                                        *')
print(f"*    L équation du modéle est : y={-theta[0]/theta[1]}*x+{- (theta[2]/theta[1])}                                                                       *")
print(f'*    L erreur de géneralisation : {accuracy(y_testt,decision_fun(theta,test_sett))}                                                                                                        *')
print(f'*    L erreur d appoximation : {accuracy(y_traint,decision_fun(theta,train_sett))}                                                                                                          *')
print('*                                                                                                                                               *')
print('*************************************************************************************************************************************************')