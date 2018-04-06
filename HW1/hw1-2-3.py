# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 10:25:52 2018

@author: zhewei
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 07:31:32 2018

@author: zhewei
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 00:10:58 2018

@author: zhewei
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 23:23:17 2018

@author: zhewei
"""

import math
import os
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def curve(n_point):
    # define the training curve
    X = np.linspace(0.01, 5, n_point).reshape(n_point, 1).astype('float32')
    y = np.abs((np.sin(5 * math.pi * X)/ (5 * math.pi * X + 0.001) )).reshape(n_point, 1)
    X, y = X.astype('float32'), y.astype('float32')
    return X, y
   
class neural_network(object):
    def __init__(self, store_path):
        # Hyperparameters
        self.lr = 0.0001
        self.batch_size = 200
        self.n_point = 1000
        self.model_path = './' + self.create_save_path(store_path)
        
        
    def train_curve(self):
        X, y = curve(self.n_point)        
        random_index = np.arange(self.n_point)
        np.random.shuffle(random_index)
        X, y = X[random_index], y[random_index]
        self.X_train = X[:self.n_point]
        self.y_train = y[:self.n_point]
    
    
    def input_pipe(self):
        with tf.name_scope('data'):
            self.x = tf.placeholder(tf.float32, shape=[None, 1], name='x_input')
            self.y = tf.placeholder(tf.float32, shape=[None, 1], name='y_value')
            
    def create_save_path(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    
    def model(self): 
        #model construction
        ###  Since we want to store parameters as one long vector, we first define our parameters as below and then
        ###  reshape it later according to each layer specification. 
        n_input = 1
        n_hidden = 30
        n_output = 1
        self.parameters = tf.Variable(tf.concat([tf.truncated_normal([n_input * n_hidden, 1], stddev=1), 
                                           tf.random_normal([n_hidden, 1], stddev=2),tf.truncated_normal([n_hidden * n_output,1], stddev=1), 
                                           tf.random_normal([n_output, 1], stddev=2)], axis=0))
        
        with tf.name_scope("hidden") as scope:
            idx_from = 0 
            weights = tf.reshape(tf.slice(self.parameters, begin=[idx_from, 0], size=[n_input*n_hidden, 1]), [n_input, n_hidden])
            idx_from = idx_from + n_input*n_hidden
            biases = tf.reshape(tf.slice(self.parameters, begin=[idx_from, 0], size=[n_hidden, 1]), [n_hidden]) # tf.Variable(tf.truncated_normal([n_hidden]))
            hidden = tf.nn.relu(tf.matmul(self.x, weights)) + biases
            
        with tf.name_scope("output_layer") as scope:
            idx_from = idx_from + n_hidden
            weights = tf.reshape(tf.slice(self.parameters, begin=[idx_from, 0], size=[n_hidden*n_output, 1]), [n_hidden, n_output])
            idx_from = idx_from + n_hidden*n_output
            biases = tf.reshape(tf.slice(self.parameters, begin=[idx_from, 0], size=[n_output, 1]), [n_output]) 
            self.y_predict = tf.nn.relu(tf.matmul(hidden, weights)) + biases
       
    def loss(self):
        #mse
        self.loss = tf.reduce_mean(tf.square(self.y_predict-self.y), name='loss')
       
    def gradient_norm(self):
        var_grad = tf.gradients(self.loss, tf.trainable_variables())
        grad_square = 0
        for arr in var_grad:
            grad_square += tf.reduce_sum(tf.reduce_sum(tf.square(arr)))
        self.grad_square = grad_square
        self.grad_norm = tf.sqrt(grad_square)
    
    def optimize_op(self):
        #Adam
        self.opt_loss = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.opt_grad = tf.train.AdamOptimizer(self.lr).minimize(self.grad_square)
        
    def summary(self):
        #Create summaries to write on TensorBoard
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()
            
    def hessian(self):
        tvars = tf.trainable_variables()
        # Get gradients of loss with repect to parameters
        dloss_dw = tf.gradients(self.loss, tvars)[0]
        dim, _ = dloss_dw.get_shape()
        hess = []
        for i in range(dim):
            # tf.slice: https://www.tensorflow.org/versions/0.6.0/api_docs/python/array_ops.html#slice
            dfx_i = tf.slice(dloss_dw, begin=[i,0] , size=[1,1])
            ddfx_i = tf.gradients(dfx_i, self.parameters)[0] # whenever we use tf.gradients, make sure you get the actual tensors by putting [0] at the end
            hess.append(ddfx_i)
        hess = tf.squeeze(hess) 
        self.hess = hess
        
    def build(self):
        self.train_curve()
        self.input_pipe()
        self.model()
        self.loss()
        self.gradient_norm()
        self.optimize_op()
        self.hessian()
        self.summary()
       
    # thre means after "thre" epoch, grad will be used as loss function   
    def train(self, n_epochs, thre):       
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            n_parameters = np.sum([np.prod(v.get_shape().as_list()) \
                                           for v in tf.trainable_variables()])
            print("Total Trainable parameters: {}".format(n_parameters))
                                   
            for epoch in range(n_epochs):  
                # use mse as loss function
                if epoch < thre:
                    for j in range(int(self.n_point/self.batch_size)):
                        start = (j * self.batch_size) % self.n_point
                        end = min(start + self.batch_size, self.n_point)
                        sess.run(self.opt_loss, feed_dict={self.x: self.X_train[start:end], 
                                                           self.y: self.y_train[start:end]})
                    
                
                # use grad as loss function     
                else:
                    # training use gradient as loss function
                    for j in range(int(self.n_point/self.batch_size)):
                        start = (j * self.batch_size) % self.n_point
                        end = min(start + self.batch_size, self.n_point)
                        sess.run(self.opt_grad, feed_dict={self.x: self.X_train[start:end], 
                                                           self.y: self.y_train[start:end]})
                                 
                # print loss every 100 epochs
                if epoch % 100 == 0:
                    v_loss, v_grad_norm = sess.run([self.loss, self.grad_norm], 
                                                   feed_dict={self.x: self.X_train,
                                                              self.y: self.y_train})
                    print("In {} epoch, grad_norm is {:.09f}, loss is {:.09f}".format(epoch, v_grad_norm, v_loss))
            
            print("Calculating Training Loss ...")
            self.tr_loss = sess.run(self.loss, feed_dict={self.x: self.X_train,
                                                          self.y: self.y_train})
            print("Training Loss = {}".format(self.tr_loss))
            
            print("Calculating Hessian Matrix ..")
            hess = sess.run(self.hess, feed_dict={self.x: self.X_train,
                                                  self.y: self.y_train})
            print("Calculating Min-Ratio ...")
            self.min_ratio(hess)
            
    
    def min_ratio(self, hessian):
        eigValue, eigVec = np.linalg.eig(hessian)
        self.min_ratio = sum(eigValue > 0) / len(eigValue)
        print("len of eigenvalue array: {}".format(len(eigValue)))
        print("num of eig > 0 : {}".format(sum(eigValue > 0)))
        print("num of eig < 0 : {}".format(len(eigValue)-sum(eigValue>0)))
        print("Min_ratio : {}".format(self.min_ratio) )
        
                  
    # write training loss and y_predict for problem 1               
    def write_loss_minRatio(self, filename):
        file_exists = os.path.isfile(filename)
        with open(filename, 'a') as fout:
            writer = csv.writer(fout, delimiter=',', lineterminator='\n')
            
            # write header at first time
            if not file_exists:
                writer.writerow(["train_loss", "Min_ratio"])
            writer.writerow([str(self.tr_loss), str(self.min_ratio)])
            
                                             
def plot_loss_minRatio(filename):
    dframe = pd.read_csv(filename, sep=',', header=0) 
    loss = np.array(dframe['train_loss'])
    min_ratio = np.array(dframe['Min_ratio']) 
    
    fig1 = plt.figure()
    plt.scatter(min_ratio, loss, color='blue')
    plt.title('Loss - Min_Ratio')
    plt.xlabel('nin_ratio')
    plt.ylabel('loss')
    plt.legend()
   
if __name__ == '__main__':
    for i in range(100):
        print("==============================")
        print("At Model {} ...".format(i + 1))
        tf.reset_default_graph() 
        model = neural_network(store_path='model')
        model.build()
        model.train(n_epochs=10000, thre=9500)
        model.write_loss_minRatio("Loss_MinRatio.csv")
        
    plot_loss_minRatio("Loss_MinRatio.csv")
    
    
    
