from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#-------- data ~ I.I.D. with noise---------------------------------------------------------------------------- #

data = pd.read_csv('Adalin_Pocket.csv')

#---- Division des données entre 80% pour train-set et 20% pour test-set

test_set = data.sample(int(len(data)*0.2),random_state=0) # 20%
train_set = data.drop(test_set.index,axis=0) # 80%


y_test = np.array(test_set['label'])
y_train = np.array(train_set['label'])

test_set.drop(['label'],axis=1,inplace=True)
train_set.drop(['label'],axis=1,inplace=True)

test_set = np.array(test_set)
train_set = np.array(train_set)


#-------------------- loss function -------------------------------------------------------------------------------
def loss(w,X,y):
    s = 0
    for i in range(len(X)):
        if np.sign(np.dot(w,np.array([X[i][0],X[i][1],1])))!=y[i]:
            s=s+1
    # print(s)
    return s/len(X)

# ------------------------------------ La phase d'entrainement ----------------------------#
def pocket(w0,X,y,Tm):
    wp = w0
    w = wp
    ctr = 0
    lp = loss(wp,X,y)
    for i in range(len(X)):
        if y[i]==-1:
            plt.scatter(X[i][0],X[i][1],color='red')
        if y[i]==1:
            plt.scatter(X[i][0],X[i][1],color='blue')
    t = np.arange(X[:,0].min()-1,X[:,0].max()+1,0.1)
    z = (-wp[0]/wp[1])*t - wp[2]/wp[1]
    plt.axis([X[:,0].min()-1,X[:,0].max()+1,X[:,1].min()-1,X[:,1].max()+1])
    plt.plot(t,z,color='green')
    plt.draw()
    plt.title('Pocket for data non_linearly separable')
    plt.pause(2)
    fig.clear()
    for k in range(Tm):

        for i in range(len(X)):
            if np.sign(np.dot(wp,np.array([X[i][0],X[i][1],1])))!=y[i]:
                wp = wp + y[i]*(np.array([X[i][0],X[i][1],1]))
                ctr = ctr+1
                for j in range(len(X)):
                    if y[j]==-1:
                        plt.scatter(X[j][0],X[j][1],color='red')
                        if i==j:
                            plt.plot(X[j][0],X[j][1],'ro',markersize = 15,alpha=.5)
                    if y[j]==1:
                        plt.scatter(X[j][0],X[j][1],color='blue')
                        if i==j:
                            plt.plot(X[j][0],X[j][1],'bo',markersize = 15,alpha=.5)
                t = np.arange(X[:,0].min()-1,X[:,0].max()+1,0.1)
                z = (-wp[0]/wp[1])*t - wp[2]/wp[1]
                plt.axis([X[:,0].min()-1,X[:,0].max()+1,X[:,1].min()-1,X[:,1].max()+1])
                plt.plot(t,z,color='green')
                plt.draw()
                plt.title('Pocket for data non_linearly separable')
                plt.pause(0.5)
                fig.clear()

        lst = loss(wp,X,y)
        if (lst<lp):
            w=wp
            lp=lst
    
    return w,lp,ctr
#-------------------------------------- Visualisation de separateur ---------------------------------------------------------------------#

fig = plt.figure()
o=np.array([0.5,0.5,0.5])
theta,ls,ctr = pocket(o,train_set,y_train,5)
print(f"Target weights : {theta}")

for i in range(len(train_set)):
    if y_train[i]==-1:
        plt.scatter(train_set[i][0],train_set[i][1],color='red')
    if y_train[i]==1:
        plt.scatter(train_set[i][0],train_set[i][1],color='blue')

t = np.arange(train_set[:,0].min()-1,train_set[:,0].max()+1,0.1)
z = (-theta[0]/theta[1])*t - (theta[2]/theta[1])
plt.axis([train_set[:,0].min()-1,train_set[:,0].max()+1,train_set[:,1].min()-1,train_set[:,1].max()+1])
plt.plot(t,z,color='k')
plt.title("Best separator")
plt.show()

#----------------------------------------------------- Resume -------------------------------------------- #
print('*********************************************** Résumé ********************************************************************')
print('*                                                                                                                          *')
print('*    Nous avons utilisé la métode sample de pandas pour génerer un échatillon aléatoire de 20%                             *')
print('*    Le VC-dimension est 2+1 = 3, Donc Pour ep = 0.5 on aura delta=0.4 pour entrainer 80% des données.                     *')
print(f"*    L équation du modéle est : y={-theta[0]/theta[1]}*x+{- (theta[2]/theta[1])}                                                                              *")
print(f'*    L erreur de géneralisation : {loss(theta,test_set,y_test)}, cette resultat est attendue car test_set est presque sans bruit.    *')
print(f'*    L erreur d appoximation : {loss(theta,train_set,y_train)}                                                                        *')
print('*    Les meilleures paramétres sélectionnées pour le modéle sont : Tmax = 5, W0 = [0.5,0.5,0.5].                           *')
print('*                                                                                                                          *')
print('***************************************************************************************************************************')