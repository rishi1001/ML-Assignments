import pandas as pd
import numpy as np
import math
import sys
import collections
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

sys.setrecursionlimit(10**6)

nodes_threshold = 5               # play with these
attribute_values = {}
attribute_values_encoded = {}
numerical_attributes = []
tot_correct_train = 0
tot_correct_test = 0
tot_correct_val = 0
nodes = []
tot_nodes = 0

prun_tot_nodes_list = []
prun_correct_train = []
prun_correct_test = []
prun_correct_val = []
prun_tot_correct_train = 0
prun_tot_correct_test = 0
prun_tot_correct_val = 0

def read(location):
    df = pd.read_csv(location,sep=';')
    data = df.to_numpy()
    X = data[:,:-1]
    Y = data[:,-1]
    Y = np.where(Y=='yes',1,0)
    Y.reshape(-1,1)
    return (X,Y)

class Node:
    isLeaf = False
    parent = None
    tot_child = 0
    attribute = None
    children = []
    data = None
    y_node = None
    data_test = None
    data_val = None
    def __init__(self, isLeaf=False , parent=None , tot_child=0 , attribute=None , children=None , data=None, y_node=None,data_test=None,data_val=None):
        self.isLeaf = isLeaf
        self.parent = parent
        self.tot_child = tot_child
        self.attribute = attribute
        self.children = children
        self.data = data
        self.y_node = y_node
        self.data_test=data_test
        self.data_val=data_val

def get_attribute_value(S):
    (X,Y)=S
    for i in range(len(X[0])):
        attribute_values[i]=[]
        for j in range(len(X)):
            if isinstance(X[j][i],int):
                attribute_values[i].append(X[j][i])
            elif X[j][i] not in attribute_values[i]:
                attribute_values[i].append(X[j][i])
        if isinstance(attribute_values[i][0],int):
            attribute_values[i].sort()

def get_numerical_attributes(S):
    (X,Y) = S
    n=0
    for i in range(len(X[0])):
        if isinstance(attribute_values[i][0],int):
            numerical_attributes.append(n)
            n+=1
        else:
            n+=len(X[0][i])
        
    return

def get_attribute_value_encoded(S):
    (X,Y)=S
    for i in range(len(X[0])):
        attribute_values_encoded[i]=[]
        for j in range(len(X)):
            if i in numerical_attributes:
                attribute_values_encoded[i].append(X[j][i])
            elif X[j][i] not in attribute_values_encoded[i]:
                attribute_values_encoded[i].append(X[j][i])
        if i in numerical_attributes:
            attribute_values_encoded[i].sort()


def select_y(S):
    (X,Y)=S
    y = np.bincount(Y).argmax()
    return y

def get_entropy(num_0,num_1,tot):
    p_0 = num_0/tot
    p_1 = num_1/tot
    if p_0==0 or p_1==0:
        return 0
    return -1*(p_0*math.log(p_0)+p_1*math.log(p_1))

def get_attribute(S):
    (X,Y) = S
    m = len(X)
    min_entropy = float('inf')
    attribute = None
    for i in range(len(X[0])):              # handle numerical and catagory differently
        if isinstance(attribute_values[i][0],int):
            tot = len(attribute_values[i])
            median = attribute_values[i][int(tot/2)]
            entropy = 0
            Y_curr = Y[np.where(X[:,i]<median)]
            if len(Y_curr)==0:
                continue
            num_1 = np.sum(Y_curr)
            num_0 = len(Y_curr)-num_1
            entropy += (len(Y_curr))*(get_entropy(num_0,num_1,len(Y_curr)))
            Y_curr = Y[np.where(X[:,i]>=median)]
            if len(Y_curr)==0:              # why do I need this
                continue
            num_1 = np.sum(Y_curr)
            num_0 = len(Y_curr)-num_1
            entropy += (len(Y_curr))*(get_entropy(num_0,num_1,len(Y_curr)))
            if entropy<min_entropy:
                min_entropy=entropy
                attribute=i
        else:
            entropy = 0
            for j in attribute_values[i]:
                isReq = X[:,i]==j
                Y_curr = Y[isReq]
                if len(Y_curr)==0:
                    continue
                # X_curr = X[isReq]
                num_1 = np.sum(Y_curr)
                num_0 = len(Y_curr)-num_1
                entropy += (len(Y_curr))*(get_entropy(num_0,num_1,len(Y_curr)))
                # exit()
            if entropy<min_entropy:
                min_entropy=entropy
                attribute=i
    return attribute

def get_attribute_encoded(S):
    (X,Y) = S
    m = len(X)
    min_entropy = float('inf')
    attribute = None
    for i in range(len(X[0])):              # handle numerical and catagory differently
        if i in numerical_attributes:
            tot = len(attribute_values_encoded[i])
            median = attribute_values_encoded[i][int(tot/2)]
            entropy = 0
            Y_curr = Y[np.where(X[:,i]<median)]
            if len(Y_curr)==0:
                continue
            num_1 = np.sum(Y_curr)
            num_0 = len(Y_curr)-num_1
            entropy += (len(Y_curr))*(get_entropy(num_0,num_1,len(Y_curr)))
            Y_curr = Y[np.where(X[:,i]>=median)]
            if len(Y_curr)==0:              # why do I need this
                continue
            num_1 = np.sum(Y_curr)
            num_0 = len(Y_curr)-num_1
            entropy += (len(Y_curr))*(get_entropy(num_0,num_1,len(Y_curr)))
            if entropy<min_entropy:
                min_entropy=entropy
                attribute=i
        else:
            entropy = 0
            for j in attribute_values_encoded[i]:
                isReq = X[:,i]==j
                Y_curr = Y[isReq]
                if len(Y_curr)==0:
                    continue
                # X_curr = X[isReq]
                num_1 = np.sum(Y_curr)
                num_0 = len(Y_curr)-num_1
                entropy += (len(Y_curr))*(get_entropy(num_0,num_1,len(Y_curr)))
                # exit()
            if entropy<min_entropy:
                min_entropy=entropy
                attribute=i
    return attribute

def get_correct(S):
    (X,Y) = S
    y = select_y(S)
    tot_1 = np.sum(Y)
    tot_0 = len(Y)-tot_1
    if y==0:
        return tot_0
    else:
        return tot_1

def get_encoded(X):
    for attribute in attribute_values:
        if not isinstance(attribute_values[attribute][0],int):
            for i in range(len(X)):
                index = attribute_values[attribute].index(X[i][attribute])
                X[i][attribute] = np.zeros(len(attribute_values[attribute]))
                X[i][attribute][index]=1
    m = len(X)
    n=0
    for i in range(len(X[0])):
        if isinstance(attribute_values[i][0],int):
            n+=1
        else:
            n+=len(X[0][i])
    X_final = np.zeros((m,n))
    for i in range(len(X)):
        curr = 0
        for j in range(len(X[0])):
            if isinstance(attribute_values[j][0],int):
                X_final[i][curr] = X[i][j]
                curr+=1 
            else:
                for val in X[i][j]:
                    X_final[i][curr] = val
                    curr+=1
        
    return X_final

def random_forest(S,S_test,S_val):
    (X,Y)=S
    X = get_encoded(X)
    max_score = -1
    best_n_estimators = None
    best_max_features = None
    best_min_samples_split = None
    best_model = None

    n_estimators_list = [50,150,250,350,450]
    max_features_list = [0.1,0.3,0.5,0.7,0.9]
    min_samples_split_list = [2,4,6,8,10]

    count = 0
    for n_estimators in n_estimators_list:
        for max_features in max_features_list:
            for min_samples_split in min_samples_split_list:
                print(count)
                count+=1
                clf = RandomForestClassifier(n_estimators=n_estimators,max_features=max_features,min_samples_split=min_samples_split,bootstrap=True,oob_score=True)
                clf.fit(X,y=Y)
                score = clf.oob_score_
                print(score)
                if score>max_score:
                    best_n_estimators=n_estimators
                    best_max_features=max_features
                    best_min_samples_split=min_samples_split
                    best_model = clf
    print(best_n_estimators,best_max_features,best_min_samples_split)
    (X_test,Y_test) = S_test
    (X_val,Y_val) = S_val
    X_test = get_encoded(X_test)
    X_val = get_encoded(X_val)
    Y_pred_train = best_model.predict(X)
    Y_pred_test = best_model.predict(X_test)
    Y_pred_val = best_model.predict(X_val)
    print(get_accuracy(Y,Y_pred_train))
    print(get_accuracy(Y_test,Y_pred_test))
    print(get_accuracy(Y_val,Y_pred_val))

    return

def solve_parameters(n_estimators,max_features,min_samples_split,X,Y):
    clf = RandomForestClassifier(n_estimators=n_estimators,max_features=max_features,min_samples_split=min_samples_split,bootstrap=True,oob_score=True)
    clf.fit(X,y=Y)
    return clf

def vary_parameters(S,S_test,S_val):

    (X,Y) = S
    X = get_encoded(X)
    (X_test,Y_test) = S_test
    (X_val,Y_val) = S_val
    X_test = get_encoded(X_test)
    X_val = get_encoded(X_val)

    n_estimators_list = [50,150,250,350,450]
    max_features_list = [0.1,0.3,0.5,0.7,0.9]
    min_samples_split_list = [2,4,6,8,10]

    n_estimators_test = []
    n_estimators_val = []

    max_features_test = []
    max_features_val = []

    min_samples_split_test = []
    min_samples_split_val = []

    count=1
    for n_estimators in n_estimators_list:
        model = solve_parameters(n_estimators,0.9,10,X,Y)
        Y_pred_test = model.predict(X_test)
        Y_pred_val = model.predict(X_val)
        test_accuracy = get_accuracy(Y_test,Y_pred_test)
        val_accuracy = get_accuracy(Y_val,Y_pred_val)
        n_estimators_test.append(test_accuracy)
        n_estimators_val.append(val_accuracy)
        print(count)
        count+=1
    
    for max_features in max_features_list:
        solve_parameters(450,max_features,10,X,Y)
        Y_pred_test = model.predict(X_test)
        Y_pred_val = model.predict(X_val)
        test_accuracy = get_accuracy(Y_test,Y_pred_test)
        val_accuracy = get_accuracy(Y_val,Y_pred_val)
        max_features_test.append(test_accuracy)
        max_features_val.append(val_accuracy)
        print(count)
        count+=1

    for min_samples_split in min_samples_split_list:
        solve_parameters(450,0.9,min_samples_split,X,Y)
        Y_pred_test = model.predict(X_test)
        Y_pred_val = model.predict(X_val)
        test_accuracy = get_accuracy(Y_test,Y_pred_test)
        val_accuracy = get_accuracy(Y_val,Y_pred_val)
        min_samples_split_test.append(test_accuracy)
        min_samples_split_val.append(val_accuracy)
        print(count)
        count+=1


    plt.plot(n_estimators_list, n_estimators_test, color='g', label='Test Set')
    plt.plot(n_estimators_list, n_estimators_val, color='m', label='Validation Set')

    plt.xlabel("Number of Estimators")
    plt.ylabel("Accuracy")
    plt.title("Variation of Number of Estimators")
    
    plt.legend()
    plt.savefig('1_E_Estimators.png')
    plt.show()
    plt.close()

    plt.plot(max_features_list, max_features_test, color='g', label='Test Set')
    plt.plot(max_features_list, max_features_val, color='m', label='Validation Set')

    plt.xlabel("Number of Max Features")
    plt.ylabel("Accuracy")
    plt.title("Variation of Max Featuers")
    
    plt.legend()
    plt.savefig('1_E_Max_Featuers.png')
    plt.show()
    plt.close()

    plt.plot(min_samples_split_list, min_samples_split_test, color='g', label='Test Set')
    plt.plot(min_samples_split_list, min_samples_split_val, color='m', label='Validation Set')

    plt.xlabel("Number of Minimum Split Samples")
    plt.ylabel("Accuracy")
    plt.title("Variation of Minimum Split Samples")
    
    plt.legend()
    plt.savefig('1_E_Min_Split_Samples.png')
    plt.show()
    plt.close()

    
    return

def count_match(y,Y):
    if len(Y)==0:
        return 0
    tot_1 = np.sum(Y)
    tot_0 = len(Y)-tot_1
    if y==0:
        return tot_0
    else:
        return tot_1


def grow_tree(S,S_test,S_val):
    global tot_correct_train,tot_correct_test,tot_correct_val,tot_nodes

    queue = collections.deque()
    node = Node(data=S,data_test=S_test,data_val=S_val)
    queue.appendleft(node)

    tot_nodes_list = []
    correct_train = []
    correct_test = []
    correct_val = []


    split_indexes = []

    tot_nodes=1
    y = select_y(S)
    node.y_node = y
    tot_correct_train += get_correct(S)
    tot_correct_test += count_match(y,node.data_test[1])
    tot_correct_val += count_match(y,node.data_val[1])
    tot_nodes_list.append(tot_nodes)
    correct_train.append(tot_correct_train)
    correct_val.append(tot_correct_val)
    correct_test.append(tot_correct_test)

    while queue:
        node = queue.pop()
        S = node.data
        S_test = node.data_test
        S_val = node.data_val
        (X,Y) = S

        (X_test,Y_test) = S_test

        (X_val,Y_val) = S_val

        tot = np.sum(Y)
        
        
        nodes.append(node)
        if len(X)<=nodes_threshold or tot==len(X) or tot==0:
            leaf = True
            # make current node as leaf
            node.isLeaf = True


            continue
        children = []
        leaf = False
        attribute = get_attribute(S)
        if isinstance(attribute_values[attribute][0],int):
            tot = len(attribute_values[attribute])
            median = attribute_values[attribute][int(tot/2)]
            X_curr = X[np.where(X[:,attribute]<median)]          # check for empty (if empty make it leaf)
            Y_curr = Y[np.where(X[:,attribute]<median)]
            if len(X_curr)!=0:                                  # handle leaf
                

                X_curr_test = X_test[np.where(X_test[:,attribute]<median)]
                Y_curr_test = Y_test[np.where(X_test[:,attribute]<median)]

                X_curr_val = X_val[np.where(X_val[:,attribute]<median)]
                Y_curr_val = Y_val[np.where(X_val[:,attribute]<median)]

                child = Node(parent=node,data=(X_curr,Y_curr),data_val=(X_curr_val,Y_curr_val),data_test=(X_curr_test,Y_curr_test))
                queue.appendleft(child)

                children.append(child)


            X_curr = X[np.where(X[:,attribute]>=median)]
            Y_curr = Y[np.where(X[:,attribute]>=median)]
            if len(X_curr)!=0:                      # handle leaf

                X_curr_test = X_test[np.where(X_test[:,attribute]>=median)]
                Y_curr_test = Y_test[np.where(X_test[:,attribute]>=median)]

                X_curr_val = X_val[np.where(X_val[:,attribute]>=median)]
                Y_curr_val = Y_val[np.where(X_val[:,attribute]>=median)]

                child = Node(parent=node,data=(X_curr,Y_curr),data_val=(X_curr_val,Y_curr_val),data_test=(X_curr_test,Y_curr_test))
                queue.appendleft(child)

                children.append(child)

        else:
            for i in attribute_values[attribute]:
                isReq = X[:,attribute]==i
                Y_curr = Y[isReq]
                X_curr = X[isReq]
                if len(X_curr)==0:
                    continue
                if len(X_curr)==len(X):
                    leaf = True
                    # make current node as leaf
                    node.isLeaf = True


                    break
                

                isReq_test = X_test[:,attribute]==i
                Y_curr_test = Y_test[isReq_test]
                X_curr_test = X_test[isReq_test]
                

                isReq_val = X_val[:,attribute]==i
                Y_curr_val = Y_val[isReq_val]
                X_curr_val = X_val[isReq_val]
                
                child = Node(parent=node,data=(X_curr,Y_curr),data_test=(X_curr_test,Y_curr_test),data_val=(X_curr_val,Y_curr_val))
                queue.appendleft(child)
                # child = grow_tree((X_curr,Y_curr))
                children.append(child)

        if len(children)==1:                # like every other child is empty
            leaf = True
            # make current node as leaf
            node.isLeaf = True
        


        if leaf==False:
            split_indexes.append(attribute)
            tot_correct_train-=get_correct(S)
            y = node.y_node
            tot_correct_test-=count_match(y,node.data_test[1])
            tot_correct_val-=count_match(y,node.data_val[1])
            for child in children:
                tot_nodes+=1
                tot_correct_train+=get_correct(child.data)
                y = select_y(child.data)
                child.y_node = y
                tot_correct_val+=count_match(y,child.data_val[1])
                tot_correct_test+=count_match(y,child.data_test[1])
            tot_nodes_list.append(tot_nodes)
            correct_train.append(tot_correct_train)
            correct_val.append(tot_correct_val)
            correct_test.append(tot_correct_test)
            node.tot_child = len(children)
            node.children = children
            node.attribute = attribute
    print("Test",tot_correct_test)
    print("Val",tot_correct_val)
    return tot_nodes_list,correct_train,correct_test,correct_val


def grow_tree_encoded(S,S_test,S_val):
    global tot_correct_train,tot_correct_test,tot_correct_val,tot_nodes

    queue = collections.deque()
    node = Node(data=S,data_test=S_test,data_val=S_val)
    queue.appendleft(node)

    tot_nodes_list = []
    correct_train = []
    correct_test = []
    correct_val = []


    split_indexes = []

    tot_nodes=1
    y = select_y(S)
    node.y_node = y
    tot_correct_train += get_correct(S)
    tot_correct_test += count_match(y,node.data_test[1])
    tot_correct_val += count_match(y,node.data_val[1])
    tot_nodes_list.append(tot_nodes)
    correct_train.append(tot_correct_train)
    correct_val.append(tot_correct_val)
    correct_test.append(tot_correct_test)

    while queue:
        node = queue.pop()
        S = node.data
        S_test = node.data_test
        S_val = node.data_val
        (X,Y) = S

        (X_test,Y_test) = S_test

        (X_val,Y_val) = S_val

        tot = np.sum(Y)
        
        
        nodes.append(node)
        if len(X)<=nodes_threshold or tot==len(X) or tot==0:
            leaf = True
            # make current node as leaf
            node.isLeaf = True


            continue
        children = []
        leaf = False
        attribute = get_attribute_encoded(S)
        if attribute in numerical_attributes:
            tot = len(attribute_values_encoded[attribute])
            median = attribute_values_encoded[attribute][int(tot/2)]
            X_curr = X[np.where(X[:,attribute]<median)]          # check for empty (if empty make it leaf)
            Y_curr = Y[np.where(X[:,attribute]<median)]
            if len(X_curr)!=0:                                  # handle leaf
                

                X_curr_test = X_test[np.where(X_test[:,attribute]<median)]
                Y_curr_test = Y_test[np.where(X_test[:,attribute]<median)]

                X_curr_val = X_val[np.where(X_val[:,attribute]<median)]
                Y_curr_val = Y_val[np.where(X_val[:,attribute]<median)]

                child = Node(parent=node,data=(X_curr,Y_curr),data_val=(X_curr_val,Y_curr_val),data_test=(X_curr_test,Y_curr_test))
                queue.appendleft(child)

                children.append(child)


            X_curr = X[np.where(X[:,attribute]>=median)]
            Y_curr = Y[np.where(X[:,attribute]>=median)]
            if len(X_curr)!=0:                      # handle leaf

                X_curr_test = X_test[np.where(X_test[:,attribute]>=median)]
                Y_curr_test = Y_test[np.where(X_test[:,attribute]>=median)]

                X_curr_val = X_val[np.where(X_val[:,attribute]>=median)]
                Y_curr_val = Y_val[np.where(X_val[:,attribute]>=median)]

                child = Node(parent=node,data=(X_curr,Y_curr),data_val=(X_curr_val,Y_curr_val),data_test=(X_curr_test,Y_curr_test))
                queue.appendleft(child)

                children.append(child)

        else:
            for i in attribute_values_encoded[attribute]:
                isReq = X[:,attribute]==i
                Y_curr = Y[isReq]
                X_curr = X[isReq]
                if len(X_curr)==0:
                    continue
                if len(X_curr)==len(X):
                    leaf = True
                    # make current node as leaf
                    node.isLeaf = True


                    break
                

                isReq_test = X_test[:,attribute]==i
                Y_curr_test = Y_test[isReq_test]
                X_curr_test = X_test[isReq_test]
                

                isReq_val = X_val[:,attribute]==i
                Y_curr_val = Y_val[isReq_val]
                X_curr_val = X_val[isReq_val]
                
                child = Node(parent=node,data=(X_curr,Y_curr),data_test=(X_curr_test,Y_curr_test),data_val=(X_curr_val,Y_curr_val))
                queue.appendleft(child)
                # child = grow_tree((X_curr,Y_curr))
                children.append(child)

        if len(children)==1:                # like every other child is empty
            leaf = True
            # make current node as leaf
            node.isLeaf = True
        


        if leaf==False:
            split_indexes.append(attribute)
            tot_correct_train-=get_correct(S)
            y = node.y_node
            tot_correct_test-=count_match(y,node.data_test[1])
            tot_correct_val-=count_match(y,node.data_val[1])
            for child in children:
                tot_nodes+=1
                tot_correct_train+=get_correct(child.data)
                y = select_y(child.data)
                child.y_node = y
                tot_correct_val+=count_match(y,child.data_val[1])
                tot_correct_test+=count_match(y,child.data_test[1])
            tot_nodes_list.append(tot_nodes)
            correct_train.append(tot_correct_train)
            correct_val.append(tot_correct_val)
            correct_test.append(tot_correct_test)
            node.tot_child = len(children)
            node.children = children
            node.attribute = attribute

    print("Test",tot_correct_test)
    print("Val",tot_correct_val)
    return tot_nodes_list,correct_train,correct_test,correct_val

def delete_node(node):
    if node.isLeaf:
        nodes.remove(node)
    else:
        for child in node.children:
            delete_node(child)
        nodes.remove(node)
    return

def pruning(node):

    global prun_tot_correct_train,prun_tot_correct_test,prun_tot_correct_val

    if node.isLeaf:
        y = node.y_node
        correct_node_train = count_match(y,node.data[1])
        correct_node_test = count_match(y,node.data_test[1])
        correct_node_val = count_match(y,node.data_val[1])
        return (correct_node_train,correct_node_test,correct_node_val)

    correct_leaf_train = 0
    correct_leaf_test = 0
    correct_leaf_val = 0
    for child in node.children:
        (train_child,test_child,val_child) = pruning(child)
        correct_leaf_train+=train_child
        correct_leaf_test+=test_child
        correct_leaf_val+=val_child
    
    y = node.y_node
    correct_node_train = count_match(y,node.data[1])
    correct_node_test = count_match(y,node.data_test[1])
    correct_node_val = count_match(y,node.data_val[1])

    if correct_node_val>=correct_leaf_val:
        for child in node.children:
            delete_node(child)

        node.children = []
        node.tot_child = 0
        node.isLeaf = True
        prun_tot_correct_train += correct_node_train - correct_leaf_train
        prun_tot_correct_test += correct_node_test - correct_leaf_test
        prun_tot_correct_val += correct_node_val - correct_leaf_val
        prun_tot_nodes_list.append(len(nodes))
        prun_correct_train.append(prun_tot_correct_train)
        prun_correct_test.append(prun_tot_correct_test)
        prun_correct_val.append(prun_tot_correct_val)
        return (correct_node_train,correct_node_test,correct_node_val)
        
    return (correct_leaf_train,correct_leaf_test,correct_leaf_val)

def get_accuracy(Y,Y_pred):
    tot = len(Y)
    correct=0
    for i in range(tot):
        if Y[i]==Y_pred[i]:
            correct+=1
    return correct/tot

def get_percent(tot_correct,m):
    return tot_correct/m

def plot(tot_nodes_list,train_accuracy_list,test_accuracy_list,val_accuracy_list,toSave):
   
    plt.plot(tot_nodes_list, train_accuracy_list, color='r', label='Train Set')
    plt.plot(tot_nodes_list, test_accuracy_list, color='g', label='Test Set')
    plt.plot(tot_nodes_list, val_accuracy_list, color='m', label='Validation Set')

    plt.xlabel("Nodes")
    plt.ylabel("Accuracy")
    plt.title("Accuracy VS Nodes")
    
    plt.legend()
    
    toSave=toSave+'.png'
    plt.savefig(toSave)
    plt.show()
    plt.close()
    return

def plot_both(tot_nodes_list,train_accuracy_list,test_accuracy_list,val_accuracy_list,prun_tot_nodes_list,prun_train_accuracy_list,prun_test_accuracy_list,prun_val_accuracy_list,toSave):
    plt.plot(tot_nodes_list, train_accuracy_list, color='r', label='Train Set')
    plt.plot(tot_nodes_list, test_accuracy_list, color='g', label='Test Set')
    plt.plot(tot_nodes_list, val_accuracy_list, color='m', label='Validation Set')

    plt.plot(prun_tot_nodes_list, prun_train_accuracy_list, color='y', label='Prunned Train Set')
    plt.plot(prun_tot_nodes_list, prun_test_accuracy_list, color='c', label='Prunned Test Set')
    plt.plot(prun_tot_nodes_list, prun_val_accuracy_list, color='k', label='Prunned Validation Set')

    plt.xlabel("Nodes")
    plt.ylabel("Accuracy")
    plt.title("Accuracy VS Nodes")
    
    plt.legend()
    
    toSave=toSave+'.png'
    plt.savefig(toSave)
    plt.show()
    plt.close()
    return


def part_A(S,S_test,S_val,m,m_test,m_val):
    global tot_correct_train,tot_correct_test,tot_correct_val,nodes,tot_nodes
    tot_nodes_list,correct_train,correct_test,correct_val = grow_tree(S,S_test,S_val)
    train_accuracy_list = [get_percent(correct,m) for correct in correct_train]
    test_accuracy_list = [get_percent(correct,m_test) for correct in correct_test]
    val_accuracy_list = [get_percent(correct,m_val) for correct in correct_val]
    plot(tot_nodes_list,train_accuracy_list,test_accuracy_list,val_accuracy_list,'1_A')
    print('Train Accuracy: ',get_percent(tot_correct_train,m))
    print('Test Accuracy: ',get_percent(tot_correct_test,m_test))
    print('Validation Accuracy: ',get_percent(tot_correct_val,m_val))
    print("Total Nodes",len(nodes))
    print("Root",nodes[0],len(nodes[0].data[0]))

    (X,Y) = S
    X_encode = get_encoded(X)
    get_attribute_value_encoded((X_encode,Y))
    (X_test,Y_test) = S_test
    X_test_encode = get_encoded(X_test)
    (X_val,Y_val) = S_val
    X_val_encode = get_encoded(X_val)

    
    tot_correct_train = 0
    tot_correct_test = 0
    tot_correct_val = 0
    nodes = []
    tot_nodes = 0

    tot_nodes_list,correct_train,correct_test,correct_val = grow_tree_encoded((X_encode,Y),(X_test_encode,Y_test),(X_val_encode,Y_val))
    train_accuracy_list = [get_percent(correct,m) for correct in correct_train]
    test_accuracy_list = [get_percent(correct,m_test) for correct in correct_test]
    val_accuracy_list = [get_percent(correct,m_val) for correct in correct_val]
    plot(tot_nodes_list,train_accuracy_list,test_accuracy_list,val_accuracy_list,'1_A_encoded')
    print('Train Accuracy: ',get_percent(tot_correct_train,m))
    print('Test Accuracy: ',get_percent(tot_correct_test,m_test))
    print('Validation Accuracy: ',get_percent(tot_correct_val,m_val))
    print("Total Nodes",len(nodes))
    print("Root",nodes[0],len(nodes[0].data[0]))
    
    return

def part_B(S,S_test,S_val,m,m_test,m_val):
    global tot_correct_train,tot_correct_test,tot_correct_val,nodes,tot_nodes,prun_tot_correct_train,prun_tot_correct_test,prun_tot_correct_val
    tot_nodes_list,correct_train,correct_test,correct_val = grow_tree(S,S_test,S_val)
    train_accuracy_list = [get_percent(correct,m) for correct in correct_train]
    test_accuracy_list = [get_percent(correct,m_test) for correct in correct_test]
    val_accuracy_list = [get_percent(correct,m_val) for correct in correct_val]

    prun_tot_nodes_list.append(len(nodes))
    prun_correct_train.append(tot_correct_train)
    prun_correct_test.append(tot_correct_test)
    prun_correct_val.append(tot_correct_val)
    prun_tot_correct_train = tot_correct_train
    prun_tot_correct_test = tot_correct_test
    prun_tot_correct_val = tot_correct_val
    print(pruning(nodes[0]))
    print("Total Nodes",len(nodes))
    print('Train Accuracy: ',get_percent(prun_tot_correct_train,m))
    print('Test Accuracy: ',get_percent(prun_tot_correct_test,m_test))
    print('Validation Accuracy: ',get_percent(prun_tot_correct_val,m_val))
    prun_train_accuracy_list = [get_percent(correct,m) for correct in prun_correct_train]
    prun_test_accuracy_list = [get_percent(correct,m_test) for correct in prun_correct_test]
    prun_val_accuracy_list = [get_percent(correct,m_val) for correct in prun_correct_val]
    plot(prun_tot_nodes_list,prun_train_accuracy_list,prun_test_accuracy_list,prun_val_accuracy_list,'1_B')
    plot_both(tot_nodes_list,train_accuracy_list,test_accuracy_list,val_accuracy_list,prun_tot_nodes_list,prun_train_accuracy_list,prun_test_accuracy_list,prun_val_accuracy_list,'1_B_both')

    return

def part_C(S,S_test,S_val):
    random_forest(S,S_test,S_val)
    
    return

def part_D(S,S_test,S_val):
    vary_parameters(S,S_test,S_val)
    return


if __name__ == "__main__":
    print('START')
    if len(sys.argv) != 5:
        sys.stderr("Wrong format for command line arguments. Follow <file.py><train><test><val><part_number> ")
    train_file=str(sys.argv[1])
    test_file=str(sys.argv[2])
    val_file=str(sys.argv[3])
    S = read(train_file)
    S_test = read(test_file)
    S_val = read(val_file)
    m = len(S[0])
    m_test = len(S_test[0])
    m_val = len(S_val[0])
    part_num=str(sys.argv[4])
    get_attribute_value(S)
    get_numerical_attributes(S)
    if part_num=='a':
        part_A(S,S_test,S_val,m,m_test,m_val)
    elif part_num=='b':
        part_B(S,S_test,S_val,m,m_test,m_val)
    elif part_num=='c':
        part_C(S,S_test,S_val)
    elif part_num=='d':
        part_D(S,S_test,S_val)

