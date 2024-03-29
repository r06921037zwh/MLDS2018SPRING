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
from itertools import zip_longest
from sklearn.manifold import TSNE

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
        W = tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=0.1), 'W')
        b = tf.Variable(tf.random_normal([1, out_dim], stddev=0.1), 'b')
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
        self.layer_rec = []
        self.first_layer_rec = []
        
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
            print("Building nn with 1 hidden layer ...")
            h1 = layer(out_dim=335, in_dim=1, inputs=self.x, scope_name='h1')
            self.y_predict = layer(out_dim=1, in_dim=335, inputs=tf.nn.relu(h1), scope_name='h2')
            
        elif self.n_layer == 3:
            print("Building nn with 3 hidden layers ...")
            h1 = layer(out_dim=16, in_dim=1, inputs=self.x, scope_name='h1')
            h2 = layer(out_dim=32, in_dim=16, inputs=tf.nn.relu(h1), scope_name='h2')
            h3 = layer(out_dim=13, in_dim=32, inputs=tf.nn.relu(h2), scope_name='h3')
            self.y_predict = layer(out_dim=1, in_dim=13, inputs=tf.nn.relu(h3), scope_name='h3')
        else:
            print("Building nn with 6 hidden layers ...")
            h1 = layer(out_dim=6, in_dim=1, inputs=self.x, scope_name='h1')
            h2 = layer(out_dim=15, in_dim=6, inputs=tf.nn.relu(h1), scope_name='h2')
            h3 = layer(out_dim=15, in_dim=15, inputs=tf.nn.relu(h2), scope_name='h3')
            h4 = layer(out_dim=15, in_dim=15, inputs=tf.nn.relu(h3), scope_name='h4')
            h5 = layer(out_dim=15, in_dim=15, inputs=tf.nn.relu(h4), scope_name='h5')
            h6 = layer(out_dim=10, in_dim=15, inputs=tf.nn.relu(h5), scope_name='h6')
            self.y_predict = layer(out_dim=1, in_dim=10, inputs=tf.nn.relu(h6), scope_name='h7')        

            
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
    
    def derive_layer(self):
        hidden_h = 'h'
        layer_weight = []
        out_first = []      # record first layer's parameters
        out_all = []        # record all layers' parameters
        
        for i in range(self.n_layer + 1):  # total n+1 hidden layers  
            hidden_name = hidden_h + str(i)
            items = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=hidden_name)
            for item in items:
                tmp = item.eval().flatten()
                layer_weight.append(list(tmp))
            # only record first layer 
            if i == 1:
                out_first = np.array(sum(layer_weight, [])) 
                
        out_all = np.array(sum(layer_weight, []))
        return  out_first, out_all
    
    def train(self, n_epochs):       
        writer = tf.summary.FileWriter(os.path.join('./graphs','/curve/'), tf.get_default_graph())
        saver = tf.train.Saver()
        
        #save model path
        directory = './model'
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            n_parameters = np.sum([np.prod(v.get_shape().as_list()) \
                                           for v in tf.trainable_variables()])
            print("Total Trainable parameters: {}".format(n_parameters))
            
            #store 0st epoch weights
            #print("Storing 0st epoch weights ... ")
            self.first_layer_rec, self.layer_rec = self.derive_layer()
            #saver.save(sess, os.path.join(directory, 'model1.ckpt')) 
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
                
                # store layer_weights every 50 epochs
                if epoch % 50 ==0:
                    first_layer, all_layer = self.derive_layer()
                    self.first_layer_rec = np.vstack((self.first_layer_rec, first_layer))
                    self.layer_rec = np.vstack((self.layer_rec, all_layer))
                    
                # print loss every 100 epochs
                if epoch % 100 == 0:
                    print("loss in {} step is {:.20f}".format(epoch ,v_loss))
                    
            # prediction 
            x_range = np.linspace(0, 1, int(self.n_point)).reshape(int(self.n_point), 1).astype('float32')
            y_pred = sess.run(self.y_predict, feed_dict={self.x: x_range})
            self.pred_rec.append(y_pred)
            self.pred_rec = list(np.array(self.pred_rec).flatten())
                                           
    def write_weight(self, all_layer, filename):
        with open(filename, 'a') as fout:
            writer = csv.writer(fout, delimiter=',', lineterminator='\n')
            if all_layer == True:
                for row in self.layer_rec:
                    writer.writerow(row)
            else:
                for row in self.first_layer_rec:
                    writer.writerow(row)

def Visualize(first_file, all_layer, n_run):
    first_weights = np.genfromtxt(first_file, delimiter=",")  
    all_weights = np.genfromtxt(all_layer, delimiter=",")  
    tsne = TSNE(n_components=2, random_state=0, verbose=1)
    transformed_weights = tsne.fit_transform(first_weights)
    
    n_rec_each = int( first_weights.shape[0] / n_run)
    label = []
    for i in range(n_run):
        tmp = np.full((n_rec_each), i+1)
        label.append(tmp)
    label = np.array(label).reshape(-1)
    
    print("plotting fisrt layer_weight after tsne ...")
    cm = plt.cm.get_cmap('RdYlGn')
    fig, ax = plt.subplots()
    for g in np.unique(label):
        i = np.where(label == g)
        ax.scatter(transformed_weights[i,0], transformed_weights[i,1], label=g, cmap=cm)
    plt.title('1st - layer')
    ax.legend()
    plt.show()
    
    tsne1 = TSNE(n_components=2, random_state=0, verbose=1)
    transformed_weights = tsne1.fit_transform(all_weights)
    print("plotting all layer_weight after tsne ...")
    cm = plt.cm.get_cmap('RdYlGn')
    fig, ax = plt.subplots()
    for g in np.unique(label):
        i = np.where(label == g)
        ax.scatter(transformed_weights[i,0], transformed_weights[i,1], label=g, cmap=cm)
    plt.title('ALL - layer')
    ax.legend()
    plt.show()                    
    
if __name__ == '__main__':
    for i in range(8):
        print("Building Model {}".format(i + 1))
        tf.reset_default_graph() 
        model = neural_network(n_layer=6)
        model.build()
        model.train(n_epochs=5000)
        model.write_weight(all_layer=False, filename='1st_layer.csv',)
        model.write_weight(all_layer=True, filename='all_layer.csv')
    Visualize('1st_layer.csv', 'all_layer.csv', n_run=8)
    
