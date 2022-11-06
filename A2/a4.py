import numpy as np
from PIL import Image
import pandas as pd

# a = np.array([[3,4,3],[3,4,3]])

# b = np.array([[1,2,3],[1,1,1]])
# c = a*b
# # c = []
# # c.append(a)
# # c.append(b)
# # a = np.concatenate((c[0],c[1]))
# #print(c)

# d = np.array([2,1])
# e = np.array([[1,1],[2,3]])
# print(d.shape)
# print(e.shape)
# print(np.einsum('i,ij->j',d,e))

# df = pd.read_csv('mnist/test.csv')
# data = df.to_numpy()
# X = data[:,:-1]
# a = X[9889].reshape(28,28).astype('uint8')
# Y = data[:,-1]
# Y = Y.reshape(-1,1)
# Y_1 = np.load('a.npy')
# print(Y.shape)
# print(Y_1.shape)
# print(a.shape)
# print(Y_1)
# m_test = len(Y_1)
# # for i in range(m_test):
# #     if Y[i]!=Y_1[i]:
# #         print(i,Y[i],Y_1[i])
# img = Image.fromarray(a)
# img.save('x.png')

import matplotlib.pyplot as plt
 
# line 1 points
x1 = np.array([1e-5,1e-3,1,5,10])
x1 = np.log10(x1)
y1 = [11.74,11.74,97.3,97.44,97.44]
# plotting the line 1 points
plt.plot(x1, y1, label = "Validation Set")
 
# line 2 points
x2 = np.array([1e-5,1e-3,1,5,10])
x2 = np.log10(x2)
y2 = [71.6,71.6,97.22,97.27,97.27]
# plotting the line 2 points
plt.plot(x2, y2, label = "Test Set")
 
# naming the x axis
plt.xlabel('log(c)')
# naming the y axis
plt.ylabel('Accuracy')
# giving a title to my graph
plt.title('Accuracy with varying log(C)')
 
# show a legend on the plot
plt.legend()
plt.savefig('Plot of C vs Accuracy.png')
 
# function to show the plot
plt.show()



