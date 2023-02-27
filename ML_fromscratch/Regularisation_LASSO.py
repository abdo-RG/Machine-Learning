import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 

# Hypothetical function h( x )
def H(X) :
    return X.dot(W) + b

# Importing dataset
df = pd.read_csv( "Salary_Data.csv" )
X = df.iloc[:, :-1].values
Y = df.iloc[:, 1].values

# Splitting dataset into train and test set
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size = 1 / 3, random_state = 0 ) 

learning_rate = 0.01     
iterations = 1000     
l1_penality = 500

# nbr of training examples, nbr of features
m, n = X.shape 

# weight initialization 
W = np.zeros(n)
b = 0

# gradient descent learning
for i in range(iterations):
    Y_pred = H(X)

    # calculate gradients       
    dW = np.zeros( n )
    for j in range( n ) : 
        if W[j] > 0 :
            dW[j] = ( - ( 2 * ( X[:, j] ).dot( Y - Y_pred ) )+ l1_penality ) / m
        else :   
            dW[j] = ( - ( 2 * ( X[:, j] ).dot( Y - Y_pred ) )- l1_penality ) / m
    db = - 2 * np.sum( Y - Y_pred ) / m

    # update weights    
    W = W - learning_rate * dW
    b = b - learning_rate * db

# Prediction on test set
Y_pred = H( X_test )
print( "Predicted values ", np.round( Y_pred[:3], 2 ) ) 
print( "Real values      ", Y_test[:3] )
print( "Trained W        ", round( W[0], 2 ) )
print( "Trained b        ", round( b, 2 ) )

# Visualization on test set  
plt.scatter( X_test, Y_test, color = 'blue' )
plt.plot( X_test, Y_pred, color = 'orange' )
plt.title( 'Salary vs Experience' )
plt.xlabel( 'Years of Experience' )
plt.ylabel( 'Salary' )
plt.show()