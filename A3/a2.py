from os import name
import sys
import pandas as pd
import numpy as np
import time
import math
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

np.random.seed(1)


attribute_values = {}
label_values = []
layers = []
type_activation = None
adaptive = False

def read(location):
    df = pd.read_csv(location,header=None)
    data = df.to_numpy()
    X = data[:,:-1]
    Y = data[:,-1]
    # Y.reshape(-1,1)
    # print(X.shape)
    # print(Y.shape)
    # print(X)
    # print(Y)
    return (X,Y)

def get_attribute_value(X,Y):
    for i in range(len(X[0])):
        attribute_values[i]=[]
        for j in range(len(X)):
            if X[j][i] not in attribute_values[i]:
                attribute_values[i].append(X[j][i])
        attribute_values[i].sort()
    # print(attribute_values)

    for i in range(len(Y)):
        if Y[i] not in label_values:
            label_values.append(Y[i])

    # print(label_values)

    

def get_encoded(X):

    m = len(X)
    n=0
    for i in range(len(X[0])):
        n+=len(attribute_values[i])
           
    X_final = np.zeros((m,n))
    for i in range(len(X)):
        curr = 0
        for j in range(len(X[0])):
            # index = attribute_values[j].index(X[i][j])
            X_final[i][curr+X[i][j]-1] = 1
            curr+=len(attribute_values[j])
            
    return X_final

def get_encoded_Y(Y):

    m = len(Y)
    n= len(label_values)
    
    print(n)
    Y_final = np.zeros((m,n))
    for i in range(len(Y)):
        Y_final[i][Y[i]] = 1
            
    return Y_final

def activation(Z):
    if type_activation=='sigmoid':
        return 1/(1+np.exp(-Z))
    elif type_activation=='relu':
        return np.maximum(0,Z)
    

def delta_activation(Z):
    if type_activation=='sigmoid':
        return Z*(1-Z)
    elif type_activation=='relu':
        return np.where(Z>0,1,0)

def forward_prop(prev_layer,W):
    # prev_layer - M*units_prev
    # curr - M*units
    # curr[i] = sumk(WikXk)
    Z = np.dot(prev_layer,np.transpose(W))
    A = activation(Z)
    return A

def back_prop(dA_next,Z_curr,W):
    dA = np.dot(dA_next,W)*delta_activation(Z_curr)
    return dA

def sgd(X,Y,r,delta,hidden_layers):          # SCHOCASTIC GRADIENT DESCENT
    global layers
    (m,n) = X.shape
    X = np.concatenate((np.ones((m,1)),X),axis=1)
    alpha = 0.1
    W = []

    tot_labels = len(label_values)
    layers = [n]+hidden_layers+[tot_labels]
    layer_result = [None for i in range(len(layers))]
    layer_delta = [None for i in range(len(layers))]

    # print(layers)

    # for i in range(1,len(layers)):
        # W.append(0.01*np.random.random((layers[i],layers[i-1]+1)))  # random
    W = [np.random.uniform(-1, 1, (k, n+1)) * math.sqrt(2/n) for (n, k) in zip(layers[:-1], layers[1:])] # he
    print(len(W))
    for i in range(len(W)):
        print(W[i].shape)
        print(layers[i+1],layers[i]+1)



    prev_loss = -1
    epoch = 1

    while True:                 # also put some max iteration

        random = np.random.permutation(m)
        X = X[random]
        Y = Y[random]

        avg_loss = 0
        # print(epoch,prev_loss)
        if adaptive==True:
            alpha=0.1/math.sqrt(epoch)
            # print('here')

        for start in range(0,m,r):
            end = start+r
            end=min(end,m)
            batch_size = end-start
            X_curr= X[start:end,:]
            Y_curr= Y[start:end,:]

            layer_result[0]=X_curr
            for i in range(1,len(layers)):
                layer_result[i]=np.concatenate((np.ones((batch_size, 1)), forward_prop(layer_result[i-1],W[i-1])), axis=1)



            layer_delta[-1]=(Y_curr-layer_result[-1][:,1:])*(delta_activation(layer_result[-1][:,1:]))/(batch_size)

            for i in range(len(layers)-2,0,-1):
                layer_delta[i]=back_prop(layer_delta[i+1],layer_result[i][:,1:],W[i][:,1:])

            for i in range(len(layers)-1):
                W[i]=W[i]+alpha*np.dot(np.transpose(layer_delta[i+1]),layer_result[i])

            avg_loss+=np.sum((Y_curr-layer_result[-1][:,1:])**2)/(2*batch_size)     
            
            
        avg_loss = avg_loss/batch_size
        
        if prev_loss==-1:
            prev_loss=avg_loss
            continue

        # print("Loss",avg_loss)

        if (abs(avg_loss-prev_loss)<delta):
            prev_loss = avg_loss
            break 
        epoch+=1
        prev_loss=avg_loss
        
    return W,prev_loss,epoch

def get_predict(W,X):
    (m,n) = X.shape
    X = np.concatenate((np.ones((m,1)),X),axis=1)
    layer_result = [None for i in range(len(layers))]
    layer_result[0]=X
    for i in range(1,len(layers)):
        layer_result[i]=np.concatenate((np.ones((m, 1)), forward_prop(layer_result[i-1],W[i-1])), axis=1)

    # print(layer_result[-1])

    Y_pred = np.argmax(layer_result[-1][:,1:],axis=1)
    # print(Y_pred.shape) 
    # print(Y_pred)
    Y_pred.tofile('Y_pred.csv', sep = ',')

    return Y_pred

def get_accuracy(Y,Y_pred):
    correct = 0
    for i in range(len(Y)):
        if Y[i]==Y_pred[i]:
            correct+=1
    return correct/len(Y)

def plot(hidden_layers,train_accuracy_list,test_accuracy_list,toSave):
   
    plt.plot(hidden_layers, train_accuracy_list, color='r', label='Train Set')
    plt.plot(hidden_layers, test_accuracy_list, color='g', label='Test Set')

    plt.xlabel("Hidden Layer Units")
    plt.ylabel("Accuracy")
    plt.title("Hidden Layer Units VS Accuracy")
    
    plt.legend()
    toSave=toSave+'.png'
    plt.savefig(toSave)
    plt.show()
    plt.close()
    return

def get_confusion_matrix(Y_pred,Y):          
    tot_labels = 10
    matrix = np.zeros((tot_labels,tot_labels),dtype=np.int)
    for i in range(len(Y)):
        matrix[int(Y[i])][int(Y_pred[i])]+=1
    return matrix


def part_A(X_encode,Y_encode):
    print('X_encode',X_encode)
    print('Y_encode',Y_encode)
    return

def part_B(X,Y,X_encode,Y_encode,X_test,X_test_encode,Y_test,Y_test_encode):
    global type_activation
    type_activation = 'sigmoid'
    start = time.time()
    W,loss,epoch = sgd(X_encode,Y_encode,r=100,delta=0.0001,hidden_layers=[5])           # change this if needed
    print('Time Taken: ',time.time()-start)
    print('Epoch: ',epoch)
    Y_pred = get_predict(W,X_encode)
    print(get_accuracy(Y,Y_pred))

    Y_test_pred = get_predict(W,X_test_encode)
    print(get_accuracy(Y_test,Y_test_pred))
    print(get_confusion_matrix(Y_test,Y_test_pred))
    
    return

def part_C(X,Y,X_encode,Y_encode,X_test,X_test_encode,Y_test,Y_test_encode):
    global type_activation
    type_activation = 'relu'
    hidden_layers = []
    accuray_train = []
    accuray_test = []
    for hidden_layer in [5,10,15,20,25]:
        start = time.time()
        W,loss,epoch = sgd(X_encode,Y_encode,r=100,delta=0.0001,hidden_layers=[hidden_layer]) 
        print('Time Taken: ',time.time()-start)
        print('Epoch: ',epoch)        
        Y_pred = get_predict(W,X_encode)
        hidden_layers.append(hidden_layer)
        acc_train = get_accuracy(Y,Y_pred)
        accuray_train.append(acc_train)
        print('Train Accuracy: ',acc_train)
        Y_test_pred = get_predict(W,X_test_encode)
        acc_test = get_accuracy(Y_test,Y_test_pred)
        print(get_confusion_matrix(Y_test,Y_test_pred))
        print('Test Accuracy: ',acc_test)
        accuray_test.append(acc_test)
        print('done')

    plot(hidden_layers,accuray_train,accuray_test,'2_C')
    
    return

def part_D(X,Y,X_encode,Y_encode,X_test,X_test_encode,Y_test,Y_test_encode):
    global type_activation,adaptive
    type_activation = 'sigmoid'
    adaptive = True
    hidden_layers = []
    accuray_train = []
    accuray_test = []
    for hidden_layer in [5,10,15,20,25]:
        start = time.time()
        W,loss,epoch = sgd(X_encode,Y_encode,r=100,delta=0.0001,hidden_layers=[hidden_layer]) 
        print('Time Taken: ',time.time()-start)
        print('Epoch: ',epoch)         
        Y_pred = get_predict(W,X_encode)
        hidden_layers.append(hidden_layer)
        acc_train = get_accuracy(Y,Y_pred)
        accuray_train.append(acc_train)
        print('Train Accuracy: ',acc_train)
        Y_test_pred = get_predict(W,X_test_encode)
        acc_test = get_accuracy(Y_test,Y_test_pred)
        print('Test Accuracy: ',acc_test)
        accuray_test.append(acc_test)
        print(get_confusion_matrix(Y_test,Y_test_pred))
        print('done')

    plot(hidden_layers,accuray_train,accuray_test,'2_D')
    
    return

def part_E(X,Y,X_encode,Y_encode,X_test,X_test_encode,Y_test,Y_test_encode):
    global type_activation,adaptive
    type_activation = 'relu'

    start = time.time()
    W,loss,epoch = sgd(X_encode,Y_encode,r=100,delta=0.0001,hidden_layers=[100,100]) 
    print('Time Taken: ',time.time()-start)
    print('Epoch: ',epoch)        
    Y_pred = get_predict(W,X_encode)
    print(get_accuracy(Y,Y_pred))
    Y_test_pred = get_predict(W,X_test_encode)
    print(get_accuracy(Y_test,Y_test_pred))
    print(get_confusion_matrix(Y_test,Y_test_pred))

    adaptive=True
    start = time.time()
    W,loss,epoch = sgd(X_encode,Y_encode,r=100,delta=0.0001,hidden_layers=[100,100])  
    print('Time Taken: ',time.time()-start)
    print('Epoch: ',epoch)         
    Y_pred = get_predict(W,X_encode)
    print(get_accuracy(Y,Y_pred))
    Y_test_pred = get_predict(W,X_test_encode)
    print(get_accuracy(Y_test,Y_test_pred))
    print(get_confusion_matrix(Y_test,Y_test_pred))
    
    return

def part_F(X_encode,Y_encode,X_test_encode,Y_test_encode):
    model = MLPClassifier(hidden_layer_sizes=(100,100),activation='relu',solver='sgd',learning_rate_init=0.01,batch_size=100,learning_rate='adaptive',max_iter=1000,random_state=2,shuffle=True)
    model.fit(X_encode,Y_encode)

    score = model.score(X_encode,Y_encode)
    print('Train: ',score)
    score = model.score(X_test_encode,Y_test_encode)
    print('Test: ',score)
    return




if __name__ == "__main__":
    print('START')
    if len(sys.argv) != 4:
        sys.stderr("Wrong format for command line arguments. Follow <file.py><train><test><part_number> ")
    train_file=str(sys.argv[1])
    (X,Y)=read(train_file)
    test_file=str(sys.argv[2])
    (X_test,Y_test) = read(test_file)
    part_num=str(sys.argv[3])
    get_attribute_value(X,Y)
    X_encode = get_encoded(X)
    Y_encode = get_encoded_Y(Y)
    X_test_encode = get_encoded(X_test)
    Y_test_encode = get_encoded_Y(Y_test)
    if part_num=='a':
        part_A(X_encode,Y_encode)
    elif part_num=='b':
        part_B(X,Y,X_encode,Y_encode,X_test,X_test_encode,Y_test,Y_test_encode)
    elif part_num=='c':
        part_C(X,Y,X_encode,Y_encode,X_test,X_test_encode,Y_test,Y_test_encode)
    elif part_num=='d':
        part_D(X,Y,X_encode,Y_encode,X_test,X_test_encode,Y_test,Y_test_encode)
    elif part_num=='e':
        part_E(X,Y,X_encode,Y_encode,X_test,X_test_encode,Y_test,Y_test_encode)
    elif part_num=='f':
        part_F(X_encode,Y_encode,X_test_encode,Y_test_encode)
