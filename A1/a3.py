import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def read():
    df = pd.read_csv('./data/q3/logisticX.csv',header=None)
    X = df.to_numpy()
    X = (X-X.mean(axis=0).reshape((1,2)))/X.std(axis=0).reshape((1,2))   # normalise
    X = np.transpose(X)
    X_0 = np.ones((1,X.shape[1]))
    X = np.vstack((X,X_0))                       # X -> (n,m) = (3,100)
    (n,m) = X.shape
    df = pd.read_csv('./data/q3/logisticY.csv',header=None)
    Y = df.to_numpy()
    Y = np.transpose(Y)
    W = np.zeros((n,1))                             # W -> (n,1) = (3,1)
    return W,X,Y,n,m

def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))

def L(W,X,Y,m):                           # LOG-LIKELYHOOD 
    Z = np.dot(W.transpose(),X)   
    A = sigmoid(Z)
    return (np.dot(Y,np.log(np.transpose(A)))+np.dot(1-Y,np.log(np.transpose(1-A))))    

def dL(W,X,Y,m):
    Z = np.dot(W.transpose(),X)
    A = sigmoid(Z)
    diff = Y - A
    diff_T = diff.transpose()
    return np.dot(X,diff_T)

def H(W,X,Y,m):                              # HESSIAN = -SUM(H(x)*(1-H(x))*x*xT)/m
    Z = np.dot(W.transpose(),X)
    A = sigmoid(Z)
    A = A*(1-A)
    return -1*np.dot(A*X,np.transpose(X))      # VECTORIZED FORM OF HESSIAN


def condition(prev_W,W,X,Y,m):              # CONVERGENCE CONDITION
    prev_L = L(prev_W,X,Y,m)
    curr_L = L(W,X,Y,m)
    print("Condition")
    print(curr_L)
    delta = 0.00000001                          # 1e-9
    if abs(curr_L-prev_L)<delta:
        return True
    else:
        return False

def newton_method(W,X,Y,m):
    while True:
        d_L = dL(W,X,Y,m)
        H_inv = np.linalg.inv(H(W,X,Y,m))
        prev_W = W
        W = W - np.dot(H_inv,d_L)
        if condition(prev_W,W,X,Y,m):
            break
    return W

def get_pred(W,X):
    Z = np.dot(W.transpose(),X)   
    A = sigmoid(Z)
    A = A.reshape(A.shape[1])
    Y_pred = np.array([0 if i<=0.5 else 1 for i in A])
    return Y_pred

def plot_data(X,Y):
    row_ix = np.where(Y == 0)
    plt.scatter(X[0,row_ix], X[1, row_ix], cmap='Paired',label=0)
    row_ix = np.where(Y == 1)
    plt.scatter(X[0,row_ix], X[1, row_ix], cmap='Paired',label=1)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.savefig('3b_Plot.png')
    plt.show()

def plot(W,X,Y,m):       
    min1, max1 = X[0, :].min()-1, X[0, :].max()+1
    min2, max2 = X[1, :].min()-1, X[1, :].max()+1
    
    x1grid = np.arange(min1, max1, 0.01)
    x2grid = np.arange(min2, max2, 0.01)
    xx, yy = np.meshgrid(x1grid, x2grid)
    
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    r3 = np.ones((len(r1), 1))
    
    grid = np.hstack((r1,r2,r3))
    yhat = get_pred(W,grid.transpose())
    
    zz = yhat.reshape(xx.shape)

    plt.contourf(xx, yy, zz, cmap='Paired')
    row_ix = np.where(Y == 0)
    plt.scatter(X[0,row_ix], X[1, row_ix], cmap='Paired',label=0)
    row_ix = np.where(Y == 1)
    plt.scatter(X[0,row_ix], X[1, row_ix], cmap='Paired',label=1)
        
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.savefig('3b_DecisionBoundary.png')
    plt.show()


# a)
W,X,Y,n,m = read()
W = newton_method(W,X,Y,m)
print("Paramters")
print(W)

# b)
print("Data Plot")
plot_data(X,Y)
print("Plot with Decision Boundary")
plot(W,X,Y,m)

