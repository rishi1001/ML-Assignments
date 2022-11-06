import numpy as np
import pandas as pd
from cvxopt import matrix
from cvxopt import solvers
from libsvm.svmutil import *
from scipy.spatial.distance import pdist 
from scipy.spatial.distance import squareform as sqf
import math
import random

def read(location):
    df = pd.read_csv(location)
    data = df.to_numpy()
    X = data[:,:-1]
    X = X/255
    #print(X)
    Y = data[:,-1]
    Y = Y.reshape(-1,1)
    print(X.shape)
    m,n = X.shape
    print(Y.shape)
    return X,Y,m

def get_matrix(A):
    return matrix(A,tc='d')


def one_one(X,Y):
    m,n = X.shape
    R=pdist(X,'sqeuclidean')
    K=np.exp(-0.05*sqf(R))
    print(K.shape)
    P=np.dot(Y,np.transpose(Y))*K
    print(P.shape)
    P = get_matrix(P)
    Q = get_matrix(-np.ones((m,1)))
    G = get_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    C = 1
    H = get_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = get_matrix(Y.reshape(1,-1))
    B = get_matrix(np.zeros(1))
    sol = solvers.qp(P,Q,G,H,A,B)
    return sol


def get_b(X,Y,sol):
    alpha = np.array(sol['x'])
    ind = (alpha>1e-4).flatten()
    sv = X[ind]
    sv_y = Y[ind]
    sv_alpha = alpha[ind]
    R=pdist(sv,'sqeuclidean')
    K=np.exp(-0.05*sqf(R))
    b = sv_y - np.sum(K*sv_alpha*sv_y,axis=0)
    b = np.sum(b)/b.size
    print("b",b)
    return b

def norm_matrix(X, Y):
    X_2 = np.einsum('ij,ij->i', X, X)
    Y_2 = np.einsum('ij,ij->i', Y, Y)
    return np.tile(Y_2, (X_2.shape[0], 1)) + np.tile(X_2, (Y_2.shape[0], 1)).T - 2 * np.matmul(X, Y.T)

def get_pred_gaussian(X,Y,sol,X_test,b):
    alpha = np.array(sol['x'])
    ind = (alpha>1e-4).flatten()
    sv = X[ind]
    tot_sv = len(sv)
    print(sv.shape)
    print(tot_sv)
    sv_y = Y[ind]
    sv_alpha = alpha[ind]
    gamma = 0.05
    Y_pred = np.einsum('i,ij->j', (sv_alpha * sv_y).reshape(tot_sv), np.exp(-gamma * norm_matrix(sv, X_test))) + b
    Y_pred = np.where(Y_pred>=0,1,-1)
    return Y_pred

def get_XY(X,Y,i,j):
    Y = Y.reshape(len(Y))
    X_curr = X[(Y==i) | (Y==j)]
    Y_curr = Y[(Y==i) | (Y==j)]
    Y_curr = np.where(Y_curr==i,1,-1).reshape(-1,1)
    return X_curr,Y_curr

def solve_multi(X,Y,X_test):
    m_test,n_test = X_test.shape
    Y_pred = np.zeros((m_test,10))
    Y_final = np.zeros(m_test)
    for i in range(10):
        for j in range(i+1,10):
            print(i,j)
            if i==j:
                continue
            X_curr,Y_curr = get_XY(X,Y,i,j)
            sol = one_one(X_curr,Y_curr)
            b = get_b(X_curr,Y_curr,sol)
            Y_curr_pred = get_pred_gaussian(X_curr,Y_curr,sol,X_test,b)
            print(Y_curr_pred)
            for k in range(len(Y_curr_pred)):                   # check this
                if Y_curr_pred[k]==1:
                    Y_pred[k][i]+=1
                else:
                    Y_pred[k][j]+=1
    Y_final = np.argmax(Y_pred,axis=1)
    print(Y_final)
    return Y_final

    

def solve_libsvm(X,Y,m,X_test,Y_test,c):
    Y = Y.reshape(m)
    problem = svm_problem(Y,X)
    # parameter = svm_parameter('-s 0 -t 0 -c 1')                  # linear kernel
    parameter = svm_parameter('-s 0 -t 2 -c {} -g 0.05'.format(c))         # guassian kernel
    model = svm_train(problem,parameter)
    m_test,n_test = X_test.shape
    p_label, p_acc, p_val = svm_predict(Y_test.reshape(m_test), X_test, model)
    return p_acc


def get_pred(W,X,b):
    Y_pred = np.dot(X,np.transpose(W))+b
    Y_pred = np.where(Y_pred>=0,1,-1)
    print(Y_pred)
    print(W.shape)
    print(X.shape)
    print(Y.shape)
    return Y_pred

def accuracy(Y_pred,Y,m):
    correct = 0
    for i in range(m):
        if Y_pred[i]==Y[i]:
            correct+=1
    return correct/m

def separate_data(location):                        # check this
    df = pd.read_csv(location)
    data = df.to_numpy()
    random.shuffle(data)
    m,n = data.shape
    size = int(m/5)
    size+=1
    i=0
    X_part = []
    Y_part = []
    while(i<m):
        start = i
        end = min(m,i+size)
        X = data[start:end,:-1]
        X = X/255
        X_part.append(X)
        #print(X)
        Y = data[start:end,-1]
        Y = Y.reshape(-1,1)
        Y_part.append(Y)
        print(X.shape)
        print(Y.shape)
        i+=size
    return X_part,Y_part

def best_param(X_part,Y_part,X,Y,X_test,Y_test):
    acc_val = []
    acc_test = []
    for c in (1e-5,1e-3,1,5,10):
        for i in range(5):
            X_train = None
            Y_train = None
            X_val = None
            Y_val = None
            start = False
            for j in range(5):
                if j==i:
                    X_val = X_part[j]
                    Y_val = Y_part[j]
                else:
                    if start==False:
                        X_train = X_part[j]
                        Y_train = Y_part[j]
                        start = True
                    else:
                        X_train = np.concatenate((X_train,X_part[j]))
                        Y_train = np.concatenate((Y_train,Y_part[j]))
            m,n = X_train.shape
            val_acc = solve_libsvm(X_train,Y_train,m,X_val,Y_val,c)
            m,n = X.shape
            test_acc = solve_libsvm(X,Y,m,X_test,Y_test,c)          # here training data is complete ? 
            acc_val.append(val_acc[0])
            acc_test.append(test_acc[0])
            print("here")
    print(acc_val)
    print(acc_test)
    return


def get_confusion_matrix(Y_pred,Y):
    tot_labels = 5
    matrix = np.zeros((tot_labels,tot_labels))
    for i in range(len(Y)):
        matrix[Y[i]-1][Y_pred[i]-1]+=1
    return matrix


X,Y,m = read('mnist/train.csv')
X_test,Y_test,m_test = read('mnist/test.csv')

# cvoxpt kC2 guassian
# Y_test_pred=solve(X,Y,X_test)
# print(accuracy(Y_test_pred,Y_test,m_test))

# libsvm
# print(solve_libsvm(X,Y,m,X_test,Y_test,1))

# validation
X_part,Y_part = separate_data('mnist/train.csv')
best_param(X_part,Y_part,X,Y,X_test,Y_test)

