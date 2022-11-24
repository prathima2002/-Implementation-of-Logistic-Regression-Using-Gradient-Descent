# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Use the standard libraries in python for Gradient Design. 
2.Upload the dataset and check any null value using .isnull() function. 
3.Declare the default values for linear regression. 
4.Calculate the loss usinng Mean Square Error. 
5.Predict the value of y. 
6.Plot the graph respect to hours and scores using scatter plot function
```

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: prathima
RegisterNumber:  212220040156
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: B S SAIHARSHITHA
RegisterNumber: 212220040139 
*/
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()

plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
*/
```

## Output:
![image](https://github.com/prathima2002/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/blob/6ca1bb4bef74942de1042c2fab5cc177d04fcdd2/WhatsApp%20Image%202022-11-24%20at%2008.20.07.jpeg)
![image](https://github.com/prathima2002/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/blob/cae7347b8d99a08f5987927dfa38c5c81d5d4d07/WhatsApp%20Image%202022-11-24%20at%2008.20.23.jpeg)
![image](https://github.com/prathima2002/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/blob/1483b3fbf129a7fd7f096e87131517d1a118d06f/WhatsApp%20Image%202022-11-24%20at%2008.20.34.jpeg)
![imaage](https://github.com/prathima2002/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/blob/35f31d83cd40ed377c7ae36d7e1552bc5a46545d/WhatsApp%20Image%202022-11-24%20at%2008.20.44.jpeg)
## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

