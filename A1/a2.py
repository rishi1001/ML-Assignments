import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

def normal_distribution(mean,variance,m):                   
    std_dev = math.sqrt(variance)
    A = np.random.normal(loc=mean,scale=std_dev,size=(1,m))
    return A

def generate():                             # SAMPLING THE DATA
    n = 3
    m = 1000000
    X0 = np.ones((1,m))
    X1 = normal_distribution(3,4,m)
    X2 = normal_distribution(-1,4,m)
    X = np.vstack((X0,X1,X2))
    W = np.array([[3],[1],[2]])
    epsilon = normal_distribution(0,2,m)
    Y = np.dot(W.transpose(),X)+epsilon
    return X,Y,n,m


def J(W,X,Y,m):                        # J is the loss function
    Z = np.dot(W.transpose(),X)
    A = Y - Z
    A_T = A.transpose()
    return np.dot(A,A_T)/(2*m)        # J = ((Y-W_T.X).((Y-W_T.X)_T))/(2*m)

def dJ(W,X,Y,m):
    Z = np.dot(W.transpose(),X)
    diff = Y - Z
    diff_T = diff.transpose()
    return -np.dot(X,diff_T)/(m)        # dJ = -(X.((Y-W_T.X)_T))/(m)

def condition(prev_J,sum_J,k,delta):
    print("Condition")
    print(abs(sum_J-prev_J)/k)
    if abs(sum_J-prev_J)/k<delta:              #  DIFF between avg of consecutive 'k' values  
        return True
    else:
        return False
    
def sgd(W,X,Y,n,m,alpha,r,count,delta):          # SCHOCASTIC GRADIENT DESCENT
    W_list = []
    W_list.append(W)
    XY = np.vstack((X,Y))
    #XY_T = np.transpose(XY)
    k = 0
    tot_iter = 0
    prev_J = -1
    sum_J = 0
    done = False
    while True:     # also put some max iteration
        #np.random.shuffle(XY_T)                         # SHUFFLING THE DATASET(THIS IS CONSUMING TIME, ALSO IT IS ALREADY SHUFFLED)
        #XY = np.transpose(XY_T)
        for start in range(0,m,r):
            end = start+r
            end=min(end,m)
            X_curr= XY[0:n,start:end]
            Y_curr= XY[n,start:end]
            d_J = dJ(W,X_curr,Y_curr,r)
            W = W - alpha*d_J
            W_list.append(W)
            k+=1
            tot_iter+=1
            sum_J += J(W,X,Y,m)
            if k==count:                            # CHECKING CONVERGENCE EVERY 'COUNT' ITERATIONS
                k=0
                if prev_J==-1:
                    prev_J = sum_J
                    sum_J = 0
                else:
                    if condition(prev_J,sum_J,count,delta):
                        done = True
                        break
                    prev_J = sum_J
                    sum_J = 0
        if done:
            break
    return W,W_list,tot_iter

def get_test():
    df = pd.read_csv('./data/q2/q2test.csv')
    XY = df.to_numpy()
    XY = np.transpose(XY)
    (n,m) = XY.shape
    X1 = XY[0]
    X2 = XY[1]
    Y = np.array([XY[2]])
    X0 = np.ones((1,m))
    X = np.vstack((X0,X1,X2))
    return X,Y

def get_pred(W,X):
    Y_pred = np.dot(W.transpose(),X)
    return Y_pred

def error(Y,Y_pred):
    return ((Y_pred - Y) ** 2).mean()/2


def plot(W_list,count):
    W_list= np.array(W_list)
    W_0 = W_list[:,0].reshape(W_list.shape[0],)
    W_1 = W_list[:,1].reshape(W_list.shape[0],)
    W_2 = W_list[:,2].reshape(W_list.shape[0],)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    iter = 0
    while iter+count<len(W_list):
        print(iter)
        ax.plot([W_0[iter],W_0[iter+count]],[W_1[iter],W_1[iter+count]],[W_2[iter],W_2[iter+count]],color='red')
        plt.pause(0.2)
        iter += count      
    ax.set_xlabel('Theta0')
    ax.set_ylabel('Theta1')
    ax.set_zlabel('Theta2')
    plt.savefig('2d.png')
    plt.show()



# a)
X,Y,n,m = generate()
W = np.zeros((n,1))

# b)
# 1) r = 1
# W,W_list,tot_iter = sgd(W,X,Y,n,m,0.001,1,1000,0.001)
# count = 100

# 2) r = 100
# W,W_list,tot_iter = sgd(W,X,Y,n,m,0.001,100,1000,0.0001)    
# count = 100

# 3) r = 10000
W,W_list,tot_iter = sgd(W,X,Y,n,m,0.001,10000,100,0.0001)   
count = 100

# 4) r = 1000000
# W,W_list,tot_iter = sgd(W,X,Y,n,m,0.001,1000000,1,0.0001)
# count = 100

# c)
print("Parameters")
print(W)
print("Total Iteration Taken Are")
print(tot_iter)

X_test,Y_test = get_test()                  # test set
W_org = np.array([[3],[1],[2]])
Y_pred = get_pred(W_org,X_test)
print("Error on Orignal Hypothesis")
print(error(Y_test,Y_pred))
Y_pred = get_pred(W,X_test)
print("Error on Learned Hypothesis")
print(error(Y_test,Y_pred))

# d)
print("Plotting the Graph")
plot(W_list,count)
 
