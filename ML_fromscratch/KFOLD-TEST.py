import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


data = pd.read_csv('Kfold.csv')

x = np.zeros((150,2))

x[:,0] = np.array(data['feature1'])
x[:,1] = np.array(data['feature2'])
y = np.array(data['label'])


w=np.array([[1],[1],[1]])
def loss(X,Y,w):
 LF=0
 def step_func(z):
  return 1.0 if (z < 0) else 0.0
 for i, x_i in enumerate(X):
    x_i = np.insert(x_i, 0, 1).reshape(-1,1)
    LF += step_func(np.dot(w.T,x_i)*Y[i])
 LF = LF*(1/(len(X)))
 return LF
def perceptron(X,Y,w):
 LF=loss(X,Y,w)
 while LF!=0:
  for i, x_i in enumerate(X):
    x_i = np.insert(x_i, 0, 1).reshape(-1,1)
    if np.dot(w.T,x_i)*Y[i] < 0:
        w= w +(Y[i]*x_i)
        x1 = [np.min(X),np.max(X)]
        m = -w[1]/w[2]
        c = -w[0]/w[2]
        x2 = m*x1 + c   
  LF=loss(X,Y,w)
  print('Loss %s' %LF)
 return x1,x2
def KFold(X,K, shuffle = True, seed = 4321):        
    """pass in the data to create train/test split for k fold"""
    # shuffle modifies indices inplace
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        rstate = np.random.RandomState(seed)
        rstate.shuffle(indices)        
    def test( n_samples, indices):
        fold_sizes = (n_samples // K) * np.ones(K, dtype = np.int)
        fold_sizes[:n_samples%K] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            test = np.zeros(n_samples, dtype = np.bool)
            test[test_indices] = True
            yield test
            current = stop
    for test in test(n_samples, indices):
        train_index = indices[np.logical_not(test)]
        # print(train_index)
        test_index = indices[test]
        yield train_index, test_index

KF=KFold(x,K=5, shuffle=True, seed=4312)
y[y==0] = -1
for train_index, test_index in KF:
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    s=plt.scatter(X_train[:,0],X_train[:,1],c=y_train)
    x1,x2=perceptron(X_train,y_train,w)
    plt.axis([np.min(x[:,0])-0.2, np.max(x[:,0])+0.2, np.min(x[:,1])-0.2, np.max(x[:,1])+0.2])
    plt.plot(x1, x2,color='Green') 
    plt.show()