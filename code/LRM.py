#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:00:48 2019

@author: 
"""

import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression_multiclass(object):
	
    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k 
        
    def fit_miniBGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch GD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

		### YOUR CODE HERE
        n_samples, n_features = X.shape
        y = np.zeros((n_samples,self.k,))
        for x_o,label in enumerate(labels):
            y[x_o,int(label)] = 1
        
        self.W = np.zeros((self.k,n_features))
        
        for i in range(self.max_iter):
            
            for ix in range(0,n_samples, batch_size):
                step = n_samples - ix if ix+batch_size > n_samples else batch_size
                g = np.mean(np.array([self._gradient(X[k], y[k]) for k in range(ix,ix+step)]), 0)
                self.W -= self.learning_rate*g
                
		### END YOUR CODE
    

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector. 

        Returns:
            _g: An array of shape [n_features, k]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        n_features,*other = _x.shape
        q = self.softmax(np.matmul(self.W, _x))
        softmax = np.reshape(q-_y,(self.k,1))
        _g = np.matmul(softmax, np.reshape(_x,(n_features,1)).T)
        return _g
		### END YOUR CODE
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.

		### YOUR CODE HERE
        softmax = np.exp(x)/np.sum(np.exp(x))
        return softmax
		### END YOUR CODE
    
    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features, k].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        predicted_proba = np.array([self.softmax(np.matmul(self.W,X[k])) for k in range(n_samples)])
        predicted_proba = np.argmax(predicted_proba, axis=1)
        return predicted_proba
		### END YOUR CODE


    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        predicted_proba = self.predict(X)
        correct = (labels == predicted_proba)
        accuracy_score = np.sum(correct)/n_samples*100
        return accuracy_score
		### END YOUR CODE

