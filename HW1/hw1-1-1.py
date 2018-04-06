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
from itertools import zip_longest

def curve(n_point):
    # define the training curve
    X = np.linspace(0.01, 1, n_point).reshape(n_point, 1).astype('float32')
    #y = np.abs((np.sin(5 * math.pi * X)/ (5 * math.pi * X + 0.001) )).reshape(n_point, 1)
    y = np.sin(3*math.pi*X)*np.cos(6*math.pi*X)
    #plt.plot(X, y)
    X, y = X.astype('float32'), y.astype('float32')
    return X, y
        
def layer(out_dim, in_dim, inputs, scope_name='fc'):
    # define full-connected layers
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        W = tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=0.1))
        b = tf.Variable(tf.random_normal([1, out_dim], stddev=0.1))
        out = tf.add(tf.matmul(inputs, W), b)     
    return out
    
class neural_network(object):
    def __init__(self, n_layer):
        # Hyperparameters
        self.lr = 0.0001
        self.keep_prob=0.75
        self.n_epoch = 5000
        self.batch_size = 200
        self.n_point = 1000
        self.n_layer = n_layer
        # rec
        self.tr_loss_rec = []
        self.pred_rec = []
        
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
    
    def model(self): 
        #model construction
        if self.n_layer == 1:
            print("Building nn with 1 hidden layer")
            h1 = layer(out_dim=335, in_dim=1, inputs=self.x, scope_name='h1')
            self.y_predict = layer(out_dim=1, in_dim=335, inputs=tf.nn.relu(h1), scope_name='o1')
            
        elif self.n_layer == 3:
            print("Building nn with 4 hidden layer")
            h1 = layer(out_dim=16, in_dim=1, inputs=self.x, scope_name='h1')
            h2 = layer(out_dim=32, in_dim=16, inputs=tf.nn.relu(h1), scope_name='h2')
            h3 = layer(out_dim=13, in_dim=32, inputs=tf.nn.relu(h2), scope_name='h3')
            self.y_predict = layer(out_dim=1, in_dim=13, inputs=tf.nn.relu(h3), scope_name='o3')
        else:
            print("Building nn with 8 hidden layer")
            h1 = layer(out_dim=6, in_dim=1, inputs=self.x, scope_name='h1')
            h2 = layer(out_dim=15, in_dim=6, inputs=tf.nn.relu(h1), scope_name='h2')
            h3 = layer(out_dim=15, in_dim=15, inputs=tf.nn.relu(h2), scope_name='h3')
            h4 = layer(out_dim=15, in_dim=15, inputs=tf.nn.relu(h3), scope_name='h4')
            h5 = layer(out_dim=15, in_dim=15, inputs=tf.nn.relu(h4), scope_name='h5')
            h6 = layer(out_dim=10, in_dim=15, inputs=tf.nn.relu(h5), scope_name='h6')
            self.y_predict = layer(out_dim=1, in_dim=10, inputs=tf.nn.relu(h6), scope_name='o1')        

    def loss(self):
        #mse
        self.loss = tf.reduce_mean(tf.square(self.y_predict-self.y), name='loss')
        
    def optimize_op(self):
        #Adam
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        
    def summary(self):
        #Create summaries to write on TensorBoard
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            self.summary_op = tf.summary.merge_all()
    
    def build(self):
        self.train_curve()
        self.input_pipe()
        self.model()
        self.loss()
        self.optimize_op()
        self.summary()
        
    def train(self, n_epochs):       
        writer = tf.summary.FileWriter(os.path.join('./graphs','/curve/'), tf.get_default_graph())
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            n_parameters = np.sum([np.prod(v.get_shape().as_list()) \
                                           for v in tf.trainable_variables()])
            print("Total Trainable parameters: {}".format(n_parameters))
            
            for epoch in range(n_epochs):
                for j in range(int(self.n_point/self.batch_size)):
                    start = (j * self.batch_size) % self.n_point
                    end = min(start + self.batch_size, self.n_point)
                    sess.run(self.opt, feed_dict={self.x: self.X_train[start:end], 
                                                  self.y: self.y_train[start:end]})
                
                v_loss, summaries= sess.run([self.loss, self.summary_op], 
                                  feed_dict={self.x: self.X_train, 
                                             self.y: self.y_train})
                
                writer.add_summary(summaries, global_step=epoch)  
                self.tr_loss_rec.append(v_loss)
                if epoch % 100 == 0:
                    print("loss in {} step is {:.20f}".format(epoch ,v_loss))
            
            # prediction 
            x_range = np.linspace(0, 1, int(self.n_point)).reshape(int(self.n_point), 1).astype('float32')
            y_pred = sess.run(self.y_predict, feed_dict={self.x: x_range})
            self.pred_rec.append(y_pred)
            self.pred_rec = list(np.array(self.pred_rec).flatten())
            
    def write_process(self, filename):
        with open(filename, 'w') as fout:
            writer = csv.writer(fout, delimiter=',', lineterminator='\n')
            writer.writerow(["train_loss", "prediction"])
            d = [self.tr_loss_rec, self.pred_rec]
            for values in zip_longest(*d):
                writer.writerow(values)                
        
def plot_result(n_point):
    dframe1 = pd.read_csv('model1.csv', sep=',', header=0)
    dframe2 = pd.read_csv('model2.csv', sep=',', header=0)
    dframe3 = pd.read_csv('model3.csv', sep=',', header=0)
    
    # plot Accuracy - epoch
    tr_loss_1 = np.array(dframe1['train_loss'])
    tr_loss_2 = np.array(dframe2['train_loss'])
    tr_loss_3 = np.array(dframe3['train_loss'])
                 
    fig1 = plt.figure()
    x = np.arange(len(tr_loss_1))
    plt.axes(yscale='log')
    plt.plot(x, tr_loss_1, color='blue', label='1-layer DNN')
    plt.plot(x, tr_loss_2, color='orange', label='3-layer DNN')
    plt.plot(x, tr_loss_3, color='green', label='6-layer DNN')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    
    # plot Prediction
    pred_1 = np.array(dframe1['prediction']).reshape(-1, 1)
    pred_2 = np.array(dframe2['prediction']).reshape(-1, 1)
    pred_3 = np.array(dframe3['prediction']).reshape(-1, 1)
    
    fig1 = plt.figure()
    x, g_truth = curve(n_point)
    plt.plot(x, g_truth, color='red', label='ground truth')
    plt.plot(x, pred_1[:n_point], color='blue', label='1-hidden DNN')
    plt.plot(x, pred_2[:n_point], color='orange', label='3-hidden DNN')
    plt.plot(x, pred_3[:n_point], color='green', label='6-hidden DNN')
    plt.title('Prediction')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    tf.reset_default_graph() 
    model1 = neural_network(n_layer=1)
    model1.build()
    model1.train(n_epochs=20000)
    model1.write_process(filename='model1.csv')
    
    tf.reset_default_graph() 
    model2 = neural_network(n_layer=3)
    model2.build()
    model2.train(n_epochs=20000)
    model2.write_process(filename='model2.csv')
    
    tf.reset_default_graph() 
    model3 = neural_network(n_layer=6)
    model3.build()
    model3.train(n_epochs=20000)
    model3.write_process(filename='model3.csv')
    
    n_point = model1.n_point
    plot_result(n_point)

