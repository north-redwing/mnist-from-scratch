#!/Users/YujiNarita/.pyenv/shims/python3
# -*- coding: utf-8 -*-

from func import cross_entropy_error, softmax
import numpy as np
from collections import OrderedDict


# Affine Layer
class Affine:
    def __init__(self, W, B):
        self.W = W
        self.B = B
        self.X = None
        self.dW = None
        self.dX = None
        self.dB = None
        
    def forward(self, X):
        self.X = X
        Z = np.dot(X, self.W) + self.B
        
        return Z
    
    def backward(self, error):
        self.dX = np.dot(error, self.W.T)
        self.dW = np.dot(self.X.T, error)
        self.dB = np.sum(error, axis=0)
        next_error = self.dX
        return next_error

# ReLU Layer
class ReLU:
    def __init__(self):
        self.mask = None
    
    def forward(self, Z):
        self.mask = (Z<=0)
        X = Z.copy()
        X[self.mask] = 0
        
        return X
    
    def backward(self, error):
        error[self.mask] = 0
        next_error = error
        
        return next_error

# SoftmaxWithLoss Layer
class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.Y = None
        self.T = None
    
    def forward(self, Z, T):
        self.T = T
        self.Y = softmax(Z)
        self.loss = cross_entropy_error(self.Y, self.T)
        
        return self.loss
    
    def backward(self):
        batch_size = self.T.shape[0]
        next_error = (self.Y-self.T) / batch_size
        
        return next_error

# Neuralnet
class Neuralnet:
    def __init__(self, input_size=28*28, neuron_num=100, output_size=10, weight_init_std=0.1, load_params=None):
        self.parameters = {}
        
        self.parameters['W1'] = weight_init_std * np.random.randn(input_size, neuron_num)
        self.parameters['B1'] = np.zeros(neuron_num)
       

        self.parameters['W2'] = weight_init_std * np.random.randn(neuron_num, output_size)
        self.parameters['B2'] = np.zeros(output_size)
        
    # ----- パラメータを読み込む場合 -----
        if load_params:
            self.parameters['W1'] = load_params['W1']
            self.parameters['B1'] = load_params['B1']
            self.parameters['W2'] = load_params['W2']
            self.parameters['B2'] = load_params['B2']

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.parameters['W1'], self.parameters['B1'])
        self.layers['ReLU'] = ReLU()
        self.layers['Affine2'] = Affine(self.parameters['W2'], self.parameters['B2'])
        
        self.lastlayer = SoftmaxWithLoss()
    
    def pred(self, X):
        for layer in self.layers.values():
            X = layer.forward(X)
        Z = X
        
        return Z

    def pred_label(self, X):
        Z = self.pred(X)
        Y = np.argmax(Z, axis=1)
        
        return Y
    
    def loss(self, X, T):
        Z = self.pred(X)

        return  self.lastlayer.forward(Z, T)
    
    def acc(self, X, T):
        Z = self.pred(X)
        Y = np.argmax(Z, axis=1)
        data_num = X.shape[0]

        if T.ndim != 1:
            T = np.argmax(T, axis=1)

        acc = np.sum(Y==T) / float(data_num)
        
        return acc
    
    def grad(self, X, T):
        # FP : Forward Propagation
        self.loss(X, T)
        
        # BP : Backward Propagation
        error  = self.lastlayer.backward()
        layers_reverse = list(self.layers.values())
        layers_reverse.reverse()
        for layer in layers_reverse:
            error = layer.backward(error)
        
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['B1'] = self.layers['Affine1'].dB
        grads['W2'] = self.layers['Affine2'].dW
        grads['B2'] = self.layers['Affine2'].dB
        
        return grads   


if __name__ == "__main__":
    pass
