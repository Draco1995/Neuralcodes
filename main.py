# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 18:00:49 2018

@author: MSI
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from time import clock
import numpy as np
from mynet import AlexNet
import tensorflow as tf

def main(args):

    
    #using random images for testing
    X = np.random.randint(0,255,size=[1200,224,224,3])
    Y = np.random.randint(0,2,size=[1200,1000])
    
    n=600
    X_train = X[0:n]
    Y_train = Y[0:n]
    X_test = X[n:]
    Y_test = Y[n:]
    
    print(len(Y_train),len(Y_train[0]))
    
    mynet = AlexNet();
    i = 0
    t0 = clock()
    while (clock() - t0) < 120:
        mynet.fit(X_train,Y_train,n_epochs=10)
        i = i + 10
    Y_pred = mynet.predict(X_test);
    print(Y_pred)

if __name__ == "__main__":
    tf.app.run()