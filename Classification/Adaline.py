from turtle import color
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#----- data ~ I.I.D. with noise -------------------------------------------------------------- #


df = pd.read_csv('Adalin_Pocket.csv')

test_set = df.sample(int(len(df)*0.2),random_state=1) # 20%
train_set = df.drop(test_set.index,axis=0) # 80%

test_set = np.array(test_set)
train_set = np.array(train_set)


#-----------------------------------------------------loss function ----------------------------------------------------
def loss(w,X):
    s = 0
    for i in range(len(X)):
        if np.sign(np.dot(w,np.array([X[i][0],X[i][1],1])))!=X[i][2]:
            s=s+1
    return np.float16(s/len(X))
# ------------------------------------ La phase d'entrainement ----------------------------#

def adaline(w0,X,Tm):
    w=w0
    ls=loss(w,X)
    ctr=0
    for i in range(len(X)):
        if X[i][2]==-1:
            plt.scatter(X[i][0],X[i][1],color='red')
        if X[i][2]==1:
            plt.scatter(X[i][0],X[i][1],color='blue')
    t = np.arange(-20,20,0.01)
    y = (-w[0]/w[1])*t - w[0]/w[2]
    plt.axis([-20,20,-20,20])
    plt.plot(t,y,color='green')
    plt.draw()
    plt.pause(2)
    fig.clear()
    for k in range(Tm):
        for i in range(len(X)):
            if (X[i][2]-np.sign(np.dot(w,np.array([X[i][0],X[i][1],1]))))!=0:
                w = w + 2*(X[i][2]-np.dot(w,np.array([X[i][0],X[i][1],1])))*(np.array([X[i][0],X[i][1],1]))
                ctr+=1
                for j in range(len(X)):
                    if X[j][2]==-1:
                        plt.scatter(X[j][0],X[j][1],color='red')
                        if i==j:
                            plt.plot(X[j][0],X[j][1],'ro',markersize = 15,alpha=.5)
                    if X[j][2]==1:
                        plt.scatter(X[j][0],X[j][1],color='blue')
                        if i==j:
                            plt.plot(X[j][0],X[j][1],'bo',markersize = 15,alpha=.5)
                t = np.arange(-20,20,0.01)
                y = (-w[0]/w[1])*t - w[2]/w[1]
                plt.axis([-20,20,-20,20])
                plt.plot(t,y,color='green')
                plt.draw()
                plt.pause(0.5)
                fig.clear()
    ls=loss(w,X)

    return w,ls,ctr

#-------------------------------------- Visualisation de separateur ---------------------------------------------------------------------#
fig = plt.figure()
w=np.array([1,0.5,0.5])
w,ls,ctr = adaline(w,train_set,8)

for i in range(len(train_set)):
    if train_set[i][2]==-1:
        plt.scatter(train_set[i][0],train_set[i][1],color='red')
    if train_set[i][2]==1:
        plt.scatter(train_set[i][0],train_set[i][1],color='blue')

t = np.arange(-20,20,0.01)
y = (-w[0]/w[1])*t - (w[2]/w[1])
plt.axis([-20,20,-20,20])
plt.plot(t,y,color='k')
plt.title("Best separator")
plt.show()

#----------------------------------------------------- Resume -------------------------------------------- #
print('*********************************************** Résumé ********************************************************************')
print('*                                                                                                                          *')
print('*    Nous avons utilisé la métode sample de pandas pour génerer un échatillon aléatoire de 20%                             *')
print('*    Le VC-dimension est 2+1 = 3, Donc Pour ep = 0.5 on aura delta=0.4 pour entrainer 80% des données.                     *')
print(f"*    L équation du modéle est : y={-w[0]/w[1]}*x+{- (w[2]/w[1])}                                                *")
print(f'*    L erreur de géneralisation : {loss(w,test_set)}, cette resultat est attendue car test_set est sans bruit.                            *')
print(f'*    L erreur d appoximation : {loss(w,train_set)}                                                                           *')
print('*    Les meilleures paramétres sélectionnées pour le modéle sont : Tmax = 8, W0 = [1,0.5,0.5].                             *')
print('*                                                                                                                          *')
print('***************************************************************************************************************************')