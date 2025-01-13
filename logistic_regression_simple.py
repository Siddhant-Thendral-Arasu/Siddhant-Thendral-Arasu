import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

%matplotlib inline
%load_ext autoreload
%autoreload 2

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T
train_set_x = train_set_x_flatten / 255.
test_set_x = test_set_x_flatten / 255.

def sig(z):   
    s = 1 / (1 + np.exp(-z))  
    return s

def initialize(dim):    
    w = np.zeros((dim,1))
    b = 0.0    
    return w, b

def prop(w, b, X, Y):
    m = X.shape[1]
    A = sigmoid(np.dot (w.T, X) + b)
    cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
    dw = np.dot(X, (A - Y).T)/m
    db = np.sum((A - Y))/m   
    cost = np.squeeze(np.array(cost))    
    grads = {"dw": dw,
             "db": db}    
    return grads, cost

def opt(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    
    costs = []
    
    for i in range(num_iterations):      
        grads, cost = propagate(w,b,X,Y)        

        dw = grads["dw"]
        db = grads["db"]        
  
        w = w - learning_rate * dw
        b = b - learning_rate * db        
        # YOUR CODE ENDS HERE
 
        if i % 100 == 0:
            costs.append(cost)
        
            # Print the cost every 100 training iterations
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

def model_N(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w,b = initialize(X_train.shape[0])
    params, grads, costs = opt(w,b,X_train,Y_train,num_iterations,learning_rate)
    return param, grads, costs
