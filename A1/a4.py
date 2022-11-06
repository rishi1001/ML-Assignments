import numpy as np
import matplotlib.pyplot as plt

def read():                                                     # READ THE DATA
    X = np.loadtxt('./data/q4/q4x.dat')
    X = (X-X.mean(axis=0).reshape((1,2)))/X.std(axis=0).reshape((1,2))      
    X = X.transpose()
    Y = np.loadtxt('./data/q4/q4y.dat',dtype=np.str_)
    Y=np.array([0 if i=='Alaska' else 1 for i in Y]).reshape((1,100))
    (n,m) = X.shape
    return X,Y,n,m

# IMPLEMENTED VECTORIZED FORM OF PHI,MEAN0,MEAN1,SIGMA,SIGMA0,SIGMA1
def Y_1(Y,m):              
    return np.sum(Y)

def Y_0(Y,m):
    return m-Y_1(Y,m)

def phi(Y,m):  
    return Y_1(Y,m)/m

def mean0(X,Y,n,m):
    Y0 = Y_0(Y,m)
    X0 = np.sum(X*(1-Y),axis=1).reshape((n,1))
    return (X0/Y0)

def mean1(X,Y,n,m):
    Y1 = Y_1(Y,m)
    X1 = np.sum(X*Y,axis=1).reshape((n,1))
    return (X1/Y1)

def cov(X,Y,n,m):
    m0 = mean0(X,Y,n,m)
    m1 = mean1(X,Y,n,m)
    A0 = (X-m0)*(1-Y)
    A1 = (X-m1)*(Y)
    A = A0+A1
    return np.dot(A,np.transpose(A))/m

def cov0(X,Y,n,m):
    m0 = mean0(X,Y,n,m)
    A = (X-m0)*(1-Y)
    Y0 = Y_0(Y,m)
    return np.dot(A,np.transpose(A))/Y0

def cov1(X,Y,n,m):
    m1 = mean1(X,Y,n,m)
    A = (X-m1)*(Y)
    Y1 = Y_1(Y,m)
    return np.dot(A,np.transpose(A))/Y1

def plot(X,Y):                             
    row_ix = np.where(Y == 0)
    plt.scatter(X[0,row_ix], X[1, row_ix], cmap='Paired',label='Alaska')
    row_ix = np.where(Y == 1)
    plt.scatter(X[0,row_ix], X[1, row_ix], cmap='Paired',label='Canada')
    plt.xlabel('Fresh-Water')
    plt.ylabel('Marine-Water')
    plt.legend()
    plt.savefig('4b_Data.png')
    plt.show()


def boundary_expression(X,p,m0,m1,c0,c1):           # BOUNDARY EXPRESSION FROM CLASS NOTES, FOR LINEAR c0=c1=c
    cov1_inverse = np.linalg.inv(c1)
    cov0_inverse = np.linalg.inv(c0)
    det0 = np.linalg.det(c0)
    det1 = np.linalg.det(c1)
    A1 = np.transpose(X-m1).dot(cov1_inverse).dot(X-m1)/2
    A2 = np.transpose(X-m0).dot(cov0_inverse).dot(X-m0)/2
    A3 = np.log(p/(1-p))
    A4 = (np.log(det1 / det0))/2
    return A1-A2-A3+A4



def boundary_linear(X,p,m0,m1,c):
    
    row_ix = np.where(Y == 0)
    plt.scatter(X[0,row_ix], X[1, row_ix], cmap='Paired',label='Alaska')
    row_ix = np.where(Y == 1)
    plt.scatter(X[0,row_ix], X[1, row_ix], cmap='Paired',label='Canada')

    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-3, 3, 100)
    x1, x2 = np.meshgrid(x1, x2)

    z = np.zeros(x1.shape)

    for i in range(0, len(z)):
        for j in range(0, len(z[0])):
            x = np.array([x1[i][j], x2[i][j]]).reshape(2, 1)
            z[i][j] = boundary_expression(x,p,m0,m1,c,c)

    plt.contour(x1, x2, z, levels=[0],colors='green')
    plt.xlabel('Fresh-Water')
    plt.ylabel('Marine-Water')
    plt.legend()
    plt.savefig('4c_Decision-Boundary.png')
    plt.show()

def boundary_linear_quad(X,p,m0,m1,c,c0,c1):      # do this
    
    row_ix = np.where(Y == 0)
    plt.scatter(X[0,row_ix], X[1, row_ix], cmap='Paired',label='Alaska')
    row_ix = np.where(Y == 1)
    plt.scatter(X[0,row_ix], X[1, row_ix], cmap='Paired',label='Canada')

    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-3, 3, 100)
    x1, x2 = np.meshgrid(x1, x2)
    z = np.zeros(x1.shape)

    for i in range(0, len(z)):
        for j in range(0, len(z[0])):
            x = np.array([x1[i][j], x2[i][j]]).reshape(2, 1)
            z[i][j] = boundary_expression(x,p,m0,m1,c,c)

    plt.contour(x1, x2, z, levels=[0],colors='green')

    z = np.zeros(x1.shape)

    for i in range(0, len(z)):
        for j in range(0, len(z[0])):
            x = np.array([x1[i][j], x2[i][j]]).reshape(2, 1)
            z[i][j] = boundary_expression(x,p,m0,m1,c0,c1)

    plt.contour(x1, x2, z, levels=[0],colors='red')
    plt.xlabel('Fresh-Water')
    plt.ylabel('Marine-Water')
    plt.legend()
    plt.savefig('4e_Decision-Boundary.png')
    plt.show()



X,Y,n,m = read()

# a)

p = phi(Y,m)
print("Phi")
print(p)

m1 = mean1(X,Y,n,m)
print("Mean1")
print(m1)

m0 = mean0(X,Y,n,m)
print("Mean0")
print(m0)

c = cov(X,Y,n,m)
print("Sigma")
print(c)



# b) 
print("Data Plot")
plot(X,Y)

# c)
print("Linear Decision Boundary")
boundary_linear(X,p,m0,m1,c)

# d)
c1 = cov1(X,Y,n,m)
print("Sigma1")
print(c1)

c0 = cov0(X,Y,n,m)
print("Sigma0")
print(c0)

# e)
print("Quadratic and Linear Decision Boundary")
boundary_linear_quad(X,p,m0,m1,c,c0,c1)   


