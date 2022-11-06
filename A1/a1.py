import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read():                                                   # reading the data 
    df = pd.read_csv('./data/q1/linearX.csv',header=None)
    X = df.to_numpy()
    X = (X-X.mean())/X.std()                                 # normalise X
    X = X.transpose()
    X_0 = np.ones((1,X.shape[1]))
    X = np.vstack((X,X_0))                                  # X -> (n,m) = (2,100)
    (n,m) = X.shape
    df = pd.read_csv('./data/q1/linearY.csv',header=None)
    Y = df.to_numpy()
    Y = Y.transpose()                                       # Y -> (1,m) = (1,100)
    W = np.zeros((n,1))                                     # W -> (n,1) = (2,1)
    return W,X,Y,n,m


def J(W,X,Y,m):                                        # J is the loss function
    Z = np.dot(W.transpose(),X)
    A = Y - Z
    A_T = A.transpose()
    return np.dot(A,A_T)/(2*m)                         # J = ((Y-W_T.X).((Y-W_T.X)_T))/(2*m)

def dJ(W,X,Y,m):
    Z = np.dot(W.transpose(),X)
    diff = Y - Z
    diff_T = diff.transpose()
    return -np.dot(X,diff_T)/(m)                      # dJ = -(X.((Y-W_T.X)_T))/(m)

def condition(prev_W,W,X,Y,m):                        # CONDITION FOR CONVERGENCE
    prev_J = J(prev_W,X,Y,m)
    curr_J = J(W,X,Y,m)
    print("Condition")
    print(curr_J)
    delta = 0.000000001                                # delta = 1e-9
    if abs(curr_J-prev_J)<delta:
        return True
    else:
        return False

def grad_descent(W,X,Y,m,alpha):                   # GRADIENT DESCENT
    W_list = []
    J_list = []
    error_list = []
    tot_iter = 0
    while True:        
        tot_iter+=1
        W_list.append(W)     
        J_list.append(J(W,X,Y,m))
        error_list.append(error(W,X,Y,m))
        d_J = dJ(W,X,Y,m)
        prev_W = W
        W = W - alpha*d_J
        if condition(prev_W,W,X,Y,m):
            break
    return W,W_list,J_list,error_list,tot_iter

def get_pred(W,X):
    Y_pred = np.dot(W.transpose(),X)
    return Y_pred

def error(W,X,Y,m):                               # RMSE ERROR
    Y_pred = get_pred(W,X)
    return np.sqrt(((Y_pred - Y) ** 2).mean())

def plot(W,X,Y,n,m):                              # HYPOTHESIS PLOT
    X0 = X[0].reshape((1,m))
    Z = np.dot(W.transpose(),X)
    plt.title('Hypothesis Plot')
    plt.scatter(X0.reshape((m,1)),Y.reshape((m,1)),color='red')
    plt.plot(X0.reshape((m,1)),Z.reshape((m,1)))
    plt.xlabel('Acidity')
    plt.ylabel('Density')
    plt.savefig('1b_Hypothesis Plot.png')
    plt.show()

def J_plot(W0,W1,X,Y,m):
    z_mesh=np.zeros(W0.shape)
    for i in range(W0.shape[0]):
        for j in range(W0.shape[1]):
            W=np.array([[W0[i][j]],[W1[i][j]]])
            z_mesh[i,j] = J(W,X,Y,m)
    return z_mesh
    

def mesh(W_list,J_list,X,Y,m,count):               # MESH for W_0,W_1,J
    W_list= np.array(W_list)
    W_0 = W_list[:,0].reshape(W_list.shape[0],)
    W_1 = W_list[:,1].reshape(W_list.shape[0],)
    J_list = np.array(J_list)[:,0].reshape(W_list.shape[0],)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')
    x0 = np.linspace(-2,2,100)      # tune this
    x1 = np.linspace(0,2.2,100)
    mesh0, mesh1 = np.meshgrid(x0, x1)
    ax.plot_surface(mesh0, mesh1, J_plot(mesh0,mesh1,X,Y,m))
    iter = 0
    while iter<len(W_list):
        print(iter)
        ax.scatter(W_0[iter],W_1[iter],J_list[iter],color='red')
        plt.pause(0.2)
        iter += count       # tune this
    ax.set_xlabel('Theta0')
    ax.set_ylabel('Theta1')
    ax.set_zlabel('Loss')
    plt.savefig('1c_Mesh.png')
    plt.show()



def error_plot(W0,W1,X,Y,m):
    z_mesh=np.zeros(W0.shape)
    for i in range(W0.shape[0]):
        for j in range(W0.shape[1]):
            W=np.array([[W0[i][j]],[W1[i][j]]])
            z_mesh[i,j] = error(W,X,Y,m)
    return z_mesh

def contour(W_list,error_list,count):      # CONTOUR for W_0,W_1,ERROR
    W_list= np.array(W_list)
    W_0 = W_list[:,0].reshape(W_list.shape[0],)
    W_1 = W_list[:,1].reshape(W_list.shape[0],)
    error_list = np.array(error_list).reshape(W_list.shape[0],)
    x0 = np.linspace(-2,2,100)  # change this
    x1 = np.linspace(-1,3,100)
    mesh0, mesh1 = np.meshgrid(x0, x1)
    plt.contour(mesh0, mesh1, error_plot(mesh0,mesh1,X,Y,m))
    iter = 0
    while iter<len(W_list):
        print(iter)
        plt.scatter(W_0[iter],W_1[iter],error_list[iter],color='red')
        plt.pause(0.2)
        iter += count       # tune this
    plt.xlabel('Theta0')
    plt.ylabel('Theta1')
    plt.savefig('1d_Contour')          # remeber to label all graphs
    plt.show()



# a)
W,X,Y,n,m=read()

# 1)
# alpha = 0.001
# count = 100

# 2)
alpha = 0.025
count = 4

# 3)
# alpha = 0.1
# count = 1

W,W_list,J_list,error_list,tot_iter = grad_descent(W,X,Y,m,alpha)
print("Parameters")
print(W)
print("Total Iterations")
print(tot_iter)
print("Final Loss")
print(J(W,X,Y,m))

# b) 
print("Hypothesis Plot")
plot(W,X,Y,n,m)

# c)
print("Mesh Animation")
mesh(W_list,J_list,X,Y,m,count)

# d)
print("Contours of Error Function")
contour(W_list,error_list,count)       







