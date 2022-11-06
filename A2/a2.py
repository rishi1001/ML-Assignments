import sys
import numpy as np
import pandas as pd
from cvxopt import matrix
from cvxopt import solvers
from scipy.spatial.distance import pdist 
from scipy.spatial.distance import squareform as sqf
from libsvm.svmutil import *
import random
import time

def read(location):
    df = pd.read_csv(location)
    data = df.to_numpy()
    X = data[:,:-1]
    X = X/255
    Y = data[:,-1]
    X = X[(Y==4) | (Y==5)]
    Y = Y[(Y==4) | (Y==5)]
    Y = (Y-4.5)*2
    Y = Y.reshape(-1,1)
    m,n = X.shape
    return X,Y,m

def read_multi(location):
    df = pd.read_csv(location)
    data = df.to_numpy()
    X = data[:,:-1]
    X = X/255
    #print(X)
    Y = data[:,-1]
    Y = Y.reshape(-1,1)
    m,n = X.shape
    return X,Y,m

def get_matrix(A):
    return matrix(A,tc='d')

def solve(X,Y,m):
    XY = X*Y
    P = get_matrix(np.dot(XY,XY.transpose()))   
    Q = get_matrix(-np.ones((m,1)))
    G = get_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    C = 1
    H = get_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = get_matrix(Y.reshape(1,-1))
    B = get_matrix(np.zeros(1))
    sol = solvers.qp(P,Q,G,H,A,B)
    return sol

def get_XY(X,Y,i,j):
    Y = Y.reshape(len(Y))
    X_curr = X[(Y==i) | (Y==j)]
    Y_curr = Y[(Y==i) | (Y==j)]
    Y_curr = np.where(Y_curr==i,1,-1).reshape(-1,1)
    return X_curr,Y_curr


def solve_libsvm(X,Y,m,X_test,Y_test,m_test,type):
    Y = Y.reshape(m)
    problem = svm_problem(Y,X)
    parameter = None
    if type=="linear":
        parameter = svm_parameter('-s 0 -t 0 -c 1')                  # linear kernel
    else:
        parameter = svm_parameter('-s 0 -t 2 -c 1 -g 0.05')         # guassian kernel
    model = svm_train(problem,parameter)
    p_label, p_acc, p_val = svm_predict(Y_test.reshape(m_test), X_test, model)
    return p_acc

def solve_libsvm_multi(X,Y,m,X_test,Y_test,c):
    Y = Y.reshape(m)
    problem = svm_problem(Y,X)
    parameter = svm_parameter('-s 0 -t 2 -c {} -g 0.05'.format(c))         # guassian kernel
    model = svm_train(problem,parameter)
    m_test,n_test = X_test.shape
    p_label, p_acc, p_val = svm_predict(Y_test.reshape(m_test), X_test, model)
    return p_label,p_acc

def solve_gaussian(X,Y):
    m,n = X.shape
    R=pdist(X,'sqeuclidean')
    K=np.exp(-0.05*sqf(R))
    P=np.dot(Y,np.transpose(Y))*K
    P = get_matrix(P)
    Q = get_matrix(-np.ones((m,1)))
    G = get_matrix(np.vstack((np.eye(m)*-1,np.eye(m))))
    C = 1
    H = get_matrix(np.hstack((np.zeros(m), np.ones(m) * C)))
    A = get_matrix(Y.reshape(1,-1))
    B = get_matrix(np.zeros(1))
    sol = solvers.qp(P,Q,G,H,A,B)
    return sol

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
            sol = solve_gaussian(X_curr,Y_curr)
            b = get_b(X_curr,Y_curr,sol)
            Y_curr_pred = get_pred_gaussian(X_curr,Y_curr,sol,X_test,b)
            for k in range(len(Y_curr_pred)):                  
                if Y_curr_pred[k]==1:
                    Y_pred[k][i]+=1
                else:
                    Y_pred[k][j]+=1
    Y_final = np.argmax(Y_pred,axis=1)
    return Y_final

def get_param(X,Y,sol):
    alpha = np.array(sol['x'])
    W = np.dot(np.transpose(Y*alpha),X)
    ind = (alpha>1e-4).flatten()
    sv = X[ind]
    sv_y = Y[ind]
    sv_alpha = alpha[ind]
    tot_sv = len(sv)
    print("Total Support Vectors :",tot_sv)
    K = np.matmul(sv,np.transpose(sv))
    b = sv_y - np.sum(K*sv_alpha*sv_y,axis=0)
    b = np.sum(b)/b.size
    return alpha,W,b

def get_pred(W,X,b):
    Y_pred = np.dot(X,np.transpose(W))+b
    Y_pred = np.where(Y_pred>=0,1,-1)
    return Y_pred

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
    print("Total Support Vectors :",tot_sv)
    sv_y = Y[ind]
    sv_alpha = alpha[ind]
    gamma = 0.05
    Y_pred = np.einsum('i,ij->j', (sv_alpha * sv_y).reshape(tot_sv), np.exp(-gamma * norm_matrix(sv, X_test))) + b
    Y_pred = np.where(Y_pred>=0,1,-1)
    return Y_pred

def accuracy(Y_pred,Y,m):
    correct = 0
    for i in range(m):
        if Y_pred[i]==Y[i]:
            correct+=1
    return correct/m

def get_confusion_matrix(Y_pred,Y):
    tot_labels = 2
    matrix = np.zeros((tot_labels,tot_labels))
    for i in range(len(Y)):
        if Y[i]==1:
            if Y_pred[i]==Y[i]:
                matrix[0][0]+=1
            else:
                matrix[0][1]+=1
        else:
            if Y_pred[i]!=Y[i]:
                matrix[1][0]+=1
            else:
                matrix[1][1]+=1
    return matrix

def separate_data(location):                        
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
        Y = data[start:end,-1]
        Y = Y.reshape(-1,1)
        Y_part.append(Y)
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
            val_lbl,val_acc = solve_libsvm_multi(X_train,Y_train,m,X_val,Y_val,c)
            m,n = X.shape
            test_lbl,test_acc = solve_libsvm_multi(X,Y,m,X_test,Y_test,c)          
            acc_val.append(val_acc[0])
            acc_test.append(test_acc[0])
    print("Validation Accuracy :",acc_val)
    print("Test Accuracy :",acc_test)
    return

def get_confusion_matrix_multi(Y_pred,Y):          
    tot_labels = 10
    matrix = np.zeros((tot_labels,tot_labels),dtype=np.int)
    for i in range(len(Y)):
        matrix[int(Y[i])][int(Y_pred[i])]+=1
    return matrix


def binary_part_A(X,Y,m,X_test,Y_test,m_test):
    start = time.time()
    sol = solve(X,Y,m)
    alpha,W,b = get_param(X,Y,sol)
    Y_pred=get_pred(W,X,b)
    print("Train Accuracy :",accuracy(Y_pred,Y,m))

    Y_pred_test = get_pred(W,X_test,b)
    print("Test Accuracy :",accuracy(Y_pred_test,Y_test,m_test))
    print("Total Time Taken :",time.time()-start)
    return

def binary_part_B(X,Y,m,X_test,Y_test,m_test):
    start = time.time()
    sol = solve_gaussian(X,Y)
    b = get_b(X,Y,sol)
    Y_pred=get_pred_gaussian(X,Y,sol,X,b)
    print("Train Accuracy :",accuracy(Y_pred,Y,m))

    Y_pred_test = get_pred_gaussian(X,Y,sol,X_test,b)
    print("Test Accuracy :",accuracy(Y_pred_test,Y_test,m_test))
    print("Total Time Taken :",time.time()-start)
    return

def binary_part_C(X,Y,m,X_test,Y_test,m_test):
    start = time.time()
    print(solve_libsvm(X,Y,m,X_test,Y_test,m_test,"linear"))
    print("Total Time Taken for Linear Using LIBSVM :",time.time()-start)
    start = time.time()
    print(solve_libsvm(X,Y,m,X_test,Y_test,m_test,"gaussian"))
    print("Total Time Taken for Gaussian Using LIBSVM :",time.time()-start)
    return

def multi_part_A(X,Y,m,X_test,Y_test,m_test):
    start = time.time()
    Y_test_pred=solve_multi(X,Y,X_test)
    print("Test Accuracy :",accuracy(Y_test_pred,Y_test,m_test))
    print("Confusion Matrix :")
    print(get_confusion_matrix_multi(Y_test_pred,Y_test))
    print("Total Time Taken for kC2 :",time.time()-start)
    return

def multi_part_B(X,Y,m,X_test,Y_test,m_test):
    start = time.time()
    Y_test_pred, Y_test_acc = solve_libsvm_multi(X,Y,m,X_test,Y_test,1)
    print(Y_test_pred.shape)
    np.save('a',Y_test_pred)
    print("Test Accuracy :",accuracy(Y_test_pred,Y_test,m_test))
    print("Confusion Matrix :")
    print(get_confusion_matrix_multi(Y_test_pred,Y_test))
    print("Total Time Taken using LIBSVM :",time.time()-start)
    return

def multi_part_C(X,Y,m,X_test,Y_test,m_test):
    start = time.time()
    Y_test_pred=solve_multi(X,Y,X_test)
    print("Test Accuracy :",accuracy(Y_test_pred,Y_test,m_test))
    print("Confusion Matrix :")
    print(get_confusion_matrix_multi(Y_test_pred,Y_test))
    print("Total Time Taken for kC2 :",time.time()-start)

    start = time.time()
    Y_test_pred, Y_test_acc = solve_libsvm_multi(X,Y,m,X_test,Y_test,1)
    print("Test Accuracy :",accuracy(Y_test_pred,Y_test,m_test))
    print("Confusion Matrix :")
    print(get_confusion_matrix_multi(Y_test_pred,Y_test))
    print("Total Time Taken using LIBSVM :",time.time()-start)
    return

def multi_part_D(train_file,X,Y,X_test,Y_test):
    X_part,Y_part = separate_data(train_file)
    best_param(X_part,Y_part,X,Y,X_test,Y_test)
    return


if __name__ == "__main__":
    print("START")
    if len(sys.argv) != 5:
        sys.stderr("Wrong format for command line arguments. Follow <file.py><train.json><test.json><binary or multi><part_number> ")
    train_file=str(sys.argv[1])
    test_file=str(sys.argv[2])
    mul_or_bin=str(sys.argv[3])
    part_num=str(sys.argv[4])
    if mul_or_bin=='1':
        X,Y,m = read(train_file)
        X_test,Y_test,m_test = read(test_file)
        if(part_num=="a"):
            binary_part_A(X,Y,m,X_test,Y_test,m_test)
        elif(part_num=="b"):
            binary_part_B(X,Y,m,X_test,Y_test,m_test)
        elif(part_num=="c"):
            binary_part_C(X,Y,m,X_test,Y_test,m_test)
    else:
        X,Y,m = read_multi(train_file)
        X_test,Y_test,m_test = read_multi(test_file)
        if(part_num=="a"):
            multi_part_A(X,Y,m,X_test,Y_test,m_test)
        elif(part_num=="b"):
            multi_part_B(X,Y,m,X_test,Y_test,m_test)
        elif(part_num=="c"):
            multi_part_C(X,Y,m,X_test,Y_test,m_test)
        elif(part_num=="d"):
            multi_part_D(train_file,X,Y,X_test,Y_test)

