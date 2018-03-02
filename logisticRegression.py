import numpy as np
import matplotlib.pyplot as plt

def Sigmoid(z):
    return 1/(1+np.exp(-z))

def LogisticCostFunc(W , b , X, y):
    n , m = X.shape
    h = Sigmoid(W.T.dot(X)+b).T
    #print("In Log: " , h)
    return (1/m)*(-y.dot(np.log(h)) - (1-y).dot(np.log(1-h)))

def GradDescentLogisticFunc(W, b, X, y, alpha = 0.00121 , eps = 1e-8, maxIter=100000):

    n , m = X.shape
    # iteration number : iter
    iter = 0
    # prev : MSE for previous iterarion
    prev = LogisticCostFunc(W, b, X, y)
    # loop till you reach convergence
    while iter<maxIter:
        iter += 1
        y_h_T = (y-Sigmoid(W.T.dot(X)+b)).T

        # updating parameters
        W -= alpha*(-1/m)* X.dot(y_h_T)
        b -= alpha*(-1/m)* np.sum(y_h_T)

        J=LogisticCostFunc(W, b, X, y)
        if(iter%10000 == 0 ):
            print("Epoch: ", int(iter/10000) , J)
        # check whether the difference between previous MSE and current MSE is
        # too small to break the loop
        #if(abs(prev-J)<eps and iter > 1000000):
        #    break
        prev=J
    return [b,W]


data = np.loadtxt('classification-data1.txt', delimiter=',')

y = data[:,-1]
X = data[:, 0:-1].T
n , m = X.shape
y = y.reshape((1,m))

W = np.zeros(n).reshape((n,1))
b = 0

print(y.shape)
print(X.shape)
print(W.shape)

np.where(data[:,-1]==1)
np.where(data[:,-1]==0)
x0_idx = np.where(data[:,-1]==0)
x1_idx = np.where(data[:,-1]==1)

b , W = GradDescentLogisticFunc(W, b, X, y)
print("Optimal Parameters: b = ", b , "\nW =" , W)

x_cord = np.linspace(np.min(data[:,0]) , np.max(data[:,0]), 5)
y_cord = -(1/W[-1])*(W[0]*x_cord + b)

plt.plot( data[x0_idx , 0] , data[x0_idx, 1] , 'ro' ,  data[x1_idx, 0] , data[x1_idx, 1], 'bx' , x_cord, y_cord)

plt.show()


