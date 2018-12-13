# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 20:00:49 2018

@author: MSI
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class AlexNet():

    def __init__(self, learning_rate=0.0002, retrain=False):
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.x = tf.placeholder("float",[None,224,224,3])
        self.y = tf.placeholder("float",[None,1000])
        self.mode = tf.placeholder(tf.bool)
        conv1 = tf.layers.conv2d(self.x,filters = 96,
                                 kernel_size = [11,11],
                                 padding = "same",
                                 strides = 4,
                                 name = "Conv1",
                                 activation=None)
        MaxPooling1 = tf.layers.MaxPooling2D(pool_size=3,
                                             strides=2,
                                             padding='valid',
                                             name="MaxPooling1")
        maxPooling1 = MaxPooling1(conv1)
        relu1 = tf.nn.relu(maxPooling1,name="Relu1")
        Conv2 = tf.layers.Conv2D(filters = 192,
                                 kernel_size = [5,5],
                                 padding = "same",
                                 strides = 1,
                                 name = "Conv2",
                                 activation=None)
        conv2 = Conv2(relu1)
        MaxPooling2 = tf.layers.MaxPooling2D(pool_size=3,
                                             strides=2,
                                             padding='valid',
                                             name="MaxPooling2")
        maxPooling2 = MaxPooling2(conv2)
        relu2 = tf.nn.relu(maxPooling2,name="Relu2")
        Conv3 = tf.layers.Conv2D(filters = 288,
                                 kernel_size = [3,3],
                                 strides = 1,
                                 name = "Conv3",
                                 padding = "same",
                                 activation=tf.nn.relu)
        conv3 = Conv3(relu2)
        Conv4 = tf.layers.Conv2D(filters = 288,
                                 kernel_size = [3,3],
                                 strides = 1,
                                 name = "Conv4",
                                 padding = "same",
                                 activation=tf.nn.relu)
        conv4 = Conv4(conv3)
        Conv5 = tf.layers.Conv2D(filters = 256,
                                 kernel_size = [3,3],
                                 strides = 1,
                                 name = "Conv5",
                                 padding = "same",
                                 activation=None)
        conv5 = Conv5(conv4)
        MaxPooling5 = tf.layers.MaxPooling2D(pool_size=3,
                                             strides=2,
                                             padding='valid',
                                             name="MaxPooling5")
        maxPooling5 = MaxPooling5(conv5)
        relu5 = tf.nn.relu(maxPooling5)
        self.print = tf.shape(conv1)
        self.print2 = tf.shape(conv2)
        self.print3 = tf.shape(conv3)
        self.print4 = tf.shape(conv4)
        self.print5 = tf.shape(conv5)
        relu5_flat = tf.reshape(relu5,[-1, 6 * 6 * 256])
        
        Dense6 = tf.layers.Dense(units=4096,
                                 activation = None)
        dense6 = Dense6(relu5_flat)
        relu6 = tf.nn.relu(dense6)
        dropout6 = tf.layers.dropout(
                inputs=relu6, training=self.mode == True)
        
        dense7 = tf.layers.dense(inputs=dropout6,
                                 units=4096,
                                 activation = None)
        relu7 = tf.nn.relu(dense7)
        dropout7 = tf.layers.dropout(
                inputs=relu7, training= self.mode==True)
        
        self.dense8 = tf.layers.dense(inputs=dropout7,
                                 units=1000,
                                 activation = None)
        self.loss_op = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y,logits=self.dense8)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        self.train_op = optimizer.minimize(loss = self.loss_op)
        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init)
        
    def fit(self,X,Y,warm_start=True,n_epochs=10):
        if warm_start==False:
            self.sess.run(self.init)
            
        for epoch in range(n_epochs):
            print(self.sess.run([self.print,self.print2,self.print3,self.print4,self.print5],feed_dict={self.x: X, self.y:Y,self.mode:True}))
            _,c = self.sess.run([self.train_op,self.loss_op],feed_dict={self.x: X, self.y:Y,self.mode:True})
            if epoch % 3 == 0:
                print("Epoch:",(epoch+1), "cost=", format(c))
        return self

    def predict_proba(self,X):
        
        test_output = None
        test_output = self.sess.run(tf.nn.softmax(self.dense8),feed_dict={self.x: X,self.mode:False})
        return test_output
        #return np.zeros((X.shape[0],self.n_labels))

    def predict(self,X):
        ''' return a matrix of predictions for X '''
        #return self.predict_proba(X)
        return (self.predict_proba(X) >= 0.5).astype(int)