# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 00:10:59 2018

@author: zhewei
"""

import os
import time 
import pandas as pd
import csv
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

## Some common use layers definition
def conv_relu(inputs, filters, k_size, stride=1, padding='SAME', scope_name='conv'):
    '''A method that does convolution + relu on inputs'''
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_channels = inputs.shape[-1]
        kernel = tf.get_variable('kernel', 
                                [k_size, k_size, in_channels, filters], 
                                initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', 
                                [filters],
                                initializer=tf.random_normal_initializer(stddev=0.1))
        conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding=padding)
    return tf.nn.relu(conv + biases, name=scope.name)

def maxpool(inputs, k_size, stride=2, padding='SAME', scope_name='pool'):
    '''A method that does max pooling on inputs'''
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool = tf.nn.max_pool(inputs, 
                            ksize=[1, k_size, k_size, 1], 
                            strides=[1, stride, stride, 1],
                            padding=padding)
    return pool

def fully_connected(inputs, out_dim, scope_name='fc'):
    '''A fully connected linear layer on inputs'''
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]
        w = tf.get_variable('weights', [in_dim, out_dim],
                            initializer=tf.truncated_normal_initializer(stddev=0.5))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.random_normal_initializer(stddev=0.5))
        out = tf.matmul(inputs, w) + b
    return out

class ConvNet(object):
    def __init__(self, cnn_layer_num, dnn_layer_num):
        self.dnn_layer_num = dnn_layer_num
        self.cnn_layer_num = cnn_layer_num
        self.lr = 0.001
        self.keep_prob = tf.constant(0.75)
        self.gstep = tf.Variable(0, dtype=tf.int32, 
                                trainable=False, name='global_step')
        self.n_classes = 10
        
        self.batch_size = 128
        self.n_train = 55000
        self.n_test = 10000
        self.n_valid = 5000
        self.n_batch = int(self.n_train /self.batch_size)
        
        self.tr_loss_rec = []
        self.tr_acc_rec = []
        self.test_loss_rec = []
        self.test_acc_rec = []
    def get_data(self):
        with tf.name_scope('data'):
            self.img = tf.placeholder(tf.float32, shape=[None, 784], name='img')
            self.label = tf.placeholder(tf.float32, shape=[None, 10], name='label')
            '''
            # read in images and labels
            train_imgs = mnist.train.images
            train_labels = mnist.train.labels
            test_imgs = mnist.test.images
            test_labels = mnist.test.labels
            valid_imgs = mnist.validation.images
            valid_labels = mnist.validation.labels
            
            # create TensorFlow Dataset objects
            tr_data = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))
            test_data = tf.data.Dataset.from_tensor_slices((test_imgs, test_labels))
            valid_data = tf.data.Dataset.from_tensor_slices((valid_imgs, valid_labels))
            
            # batch input
            tr_data = tr_data.batch(self.batch_size)
            test_data = test_data.batch(self.batch_size)
            valid_data = valid_data.batch(self.n_valid)
            
            # create TensorFlow Iterator object
            iterator = tf.data.Iterator.from_structure(tr_data.output_types,
                                                       tr_data.output_shapes)
            img, self.label = iterator.get_next()
            self.img = tf.reshape(img, shape=[-1, 28, 28, 1])
            
            # create two initialization ops to switch between the datasets
            self.train_init = iterator.make_initializer(tr_data)
            self.test_init = iterator.make_initializer(test_data)
            self.valid_init = iterator.make_initializer(valid_data)
            '''
    def inference(self):
        img = tf.reshape(self.img, shape=[-1, 28, 28, 1])
        if self.cnn_layer_num == 1:
            print("Building 1-layer CNN  ...")
            conv1 = conv_relu(inputs=img, filters=10, k_size=5, scope_name='conv1')
            pool1 = maxpool(conv1, k_size=2, stride=2, scope_name='pool1')
            
            feature_dim = pool1.shape[1] * pool1.shape[2] * pool1.shape[3]
            cnn_out = tf.reshape(pool1, [-1, feature_dim])
            pass
        elif self.cnn_layer_num == 2:
            print("Building 2-layer CNN  ...")
            conv1 = conv_relu(inputs=img, filters=10, k_size=5, scope_name='conv1')
            pool1 = maxpool(conv1, k_size=2, stride=2, scope_name='pool1')
            
            conv2 = conv_relu(inputs=pool1, filters=36, k_size=5,scope_name='conv2')
            pool2 = maxpool(conv2, k_size=2, stride=2, scope_name='pool2') 
            
            feature_dim = pool2.shape[1] * pool2.shape[2] * pool2.shape[3]
            cnn_out = tf.reshape(pool2, [-1, feature_dim])
            
        else:
            print("Building 4-layer CNN  ...")
            conv1 = conv_relu(inputs=img, filters=10, k_size=5, scope_name='conv1')
            pool1 = maxpool(conv1, k_size=2, stride=2, scope_name='pool1')
            
            conv2 = conv_relu(inputs=pool1, filters=39, k_size=5,scope_name='conv2')
            pool2 = maxpool(conv2, k_size=2, stride=2, scope_name='pool2') 
            
            conv3 = conv_relu(inputs=pool2, filters=38, k_size=5,scope_name='conv3')
            pool3 = maxpool(conv3, k_size=2, stride=2, scope_name='pool3') 
            
            conv4 = conv_relu(inputs=pool3, filters=38, k_size=5,scope_name='conv4')
            pool4 = maxpool(conv4, k_size=2, stride=2, scope_name='pool4') 
                    
            feature_dim = pool4.shape[1] * pool4.shape[2] * pool4.shape[3]
            cnn_out = tf.reshape(pool4, [-1, feature_dim])
        
        
        if self.dnn_layer_num == 1:
            print("Building 1-Layer DNN ...")
            
            fc1 = fully_connected(inputs=cnn_out, out_dim=46, scope_name='fc1')
            dropout1 = tf.nn.dropout(tf.nn.relu(fc1), self.keep_prob, name='fc1_dropout')
                          
            self.logits = fully_connected(dropout1, self.n_classes, 'logits')
            
        elif self.dnn_layer_num == 2:
            print("Building 2-Layer DNN ...")
            
            fc1 = fully_connected(inputs=cnn_out, out_dim=36, scope_name='fc1')
            dropout1 = tf.nn.dropout(tf.nn.relu(fc1), self.keep_prob, name='fc1_dropout')
        
            fc2 = fully_connected(inputs=dropout1, out_dim=36, scope_name='fc2')
            dropout2 = tf.nn.dropout(tf.nn.relu(fc2), self.keep_prob, name='fc2_dropout')
            
            fc3 = fully_connected(inputs=dropout2, out_dim=36, scope_name='fc3')
            dropout3 = tf.nn.dropout(tf.nn.relu(fc3), self.keep_prob, name='fc3_dropout')
        
            fc4 = fully_connected(inputs=dropout3, out_dim=36, scope_name='fc4')
            dropout4 = tf.nn.dropout(tf.nn.relu(fc4), self.keep_prob, name='fc4_dropout')
               
            self.logits = fully_connected(dropout4, self.n_classes, 'logits')
            
        else:
            print("Building 4-Layer DNN ...")
            
            fc1 = fully_connected(inputs=cnn_out, out_dim=30, scope_name='fc1')
            dropout1 = tf.nn.dropout(tf.nn.relu(fc1), self.keep_prob, name='fc1_dropout')
        
            fc2 = fully_connected(inputs=dropout1, out_dim=30, scope_name='fc2')
            dropout2 = tf.nn.dropout(tf.nn.relu(fc2), self.keep_prob, name='fc2_dropout')
        
            fc3 = fully_connected(inputs=dropout2, out_dim=30, scope_name='fc3')
            dropout3 = tf.nn.dropout(tf.nn.relu(fc3), self.keep_prob, name='fc3_dropout')
        
            fc4 = fully_connected(inputs=dropout3, out_dim=30, scope_name='fc4')
            dropout4 = tf.nn.dropout(tf.nn.relu(fc4), self.keep_prob, name='fc4_dropout') 
            
            fc5 = fully_connected(inputs=dropout4, out_dim=30, scope_name='fc5')
            dropout5 = tf.nn.dropout(tf.nn.relu(fc5), self.keep_prob, name='fc5_dropout')
        
            fc6 = fully_connected(inputs=dropout5, out_dim=30, scope_name='fc6')
            dropout6 = tf.nn.dropout(tf.nn.relu(fc6), self.keep_prob, name='fc6_dropout')
        
            fc7 = fully_connected(inputs=dropout6, out_dim=30, scope_name='fc7')
            dropout7 = tf.nn.dropout(tf.nn.relu(fc7), self.keep_prob, name='fc7_dropout')
        
            fc8 = fully_connected(inputs=dropout7, out_dim=30, scope_name='fc8')
            dropout8 = tf.nn.dropout(tf.nn.relu(fc8), self.keep_prob, name='fc8_dropout') 
        
            self.logits = fully_connected(dropout8, self.n_classes, 'logits')
            
            
    def loss(self):
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name='loss')
    
    def optimize(self):
        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss, 
                                                global_step=self.gstep)

    def summary(self):
        '''Create summaries to write on TensorBoard'''
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram_loss', self.loss)
            self.summary_op = tf.summary.merge_all()
    
    def evaluate(self):
        '''Count the number of right predictions in a batch'''
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
            
    def build(self):
        '''Build the computation graph'''
        self.get_data()
        self.inference()
        self.loss()
        self.optimize()
        self.evaluate()
        self.summary()
        
    def train(self, n_epochs):       
        writer = tf.summary.FileWriter(os.path.join('./graphs/',str(self.cnn_layer_num)+ str("_") +
                                                    str(self.dnn_layer_num)), tf.get_default_graph())
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            n_parameters = np.sum([np.prod(v.get_shape().as_list()) \
                                   for v in tf.trainable_variables()])
            print("Total Trainable parameters: {}".format(n_parameters))
            
            step = self.gstep.eval()
            for epoch in range(n_epochs):
                print("================================================================")
                print("At epoch {0} :".format(epoch + 1))
                
                start_time = time.time()   
                for _ in range(self.n_batch):
                    batch_x, batch_y = mnist.train.next_batch(self.batch_size)
                    sess.run(self.opt, feed_dict={self.img: batch_x,
                                                  self.label: batch_y})
    
                tr_loss, tr_acc = sess.run([self.loss, self.accuracy],
                                     feed_dict={self.img: mnist.train.images,
                                                self.label: mnist.train.labels})
    
                loss, acc, summaries = sess.run([self.loss, self.accuracy, self.summary_op],
                                     feed_dict={self.img: mnist.test.images,
                                                self.label: mnist.test.labels})
                step += 1
                writer.add_summary(summaries, global_step=step)    
                self.tr_acc_rec.append(tr_acc)
                self.tr_loss_rec.append(tr_loss)
                self.test_acc_rec.append(acc)
                self.test_loss_rec.append(loss)
                
                print('Training Loss : {0:.04f}'.format(tr_loss))
                print('Training Accuracy : {0:.04f} '.format(tr_acc))
                print('Testing Loss : {0:.04f}'.format(loss))
                print('Testing Accuracy : {0:.04f} '.format(acc), end='')
                print(', Took: {0:.02f} seconds'.format(time.time() - start_time))
                               
            writer.close()
                             
    def write_process(self, filename):
        with open(filename, 'w') as fout:
            writer = csv.writer(fout, delimiter=',', lineterminator='\n')
            writer.writerow(["train_acc","train_loss", "test_acc", "test_loss"])
            for i in range(len(self.tr_acc_rec)):
                writer.writerow([str(self.tr_acc_rec[i]),str(self.tr_loss_rec[i]),
                                str(self.test_acc_rec[i]),str(self.test_loss_rec[i])])
        


def plot_result():
    dframe1 = pd.read_csv('model1.csv', sep=',', header=0)
    dframe2 = pd.read_csv('model2.csv', sep=',', header=0)
    dframe3 = pd.read_csv('model3.csv', sep=',', header=0)
    
    tr_acc_1 = np.array(dframe1['train_acc'])
    tr_loss_1 = np.array(dframe1['train_loss'])
       
    tr_acc_2 = np.array(dframe2['train_acc'])
    tr_loss_2 = np.array(dframe2['train_loss'])
        
    tr_acc_3 = np.array(dframe3['train_acc'])
    tr_loss_3 = np.array(dframe3['train_loss'])
    
    # plot Accuracy - epoch    
    fig1 = plt.figure()
    x = np.arange(len(tr_acc_1))
    plt.plot(x, tr_acc_1, color='blue', label='1-layer CNN')
    plt.plot(x, tr_acc_2, color='orange', label='2-layer CNN' )
    plt.plot(x, tr_acc_3, color='red', label='4-layer CNN')
    plt.title('Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()
    plt.show()
    
    # plot Loss - epoch
    fig2 = plt.figure()
    x = np.arange(len(tr_loss_1))
    plt.plot(x, tr_loss_1, color='blue', label='1-layer CNN')
    plt.plot(x, tr_loss_2, color='orange', label='2-layer CNN' )
    plt.plot(x, tr_loss_3, color='red', label='4-layer CNN')
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
    
    
if __name__ == '__main__':
    tf.reset_default_graph() 
    model1 = ConvNet(cnn_layer_num=1, dnn_layer_num=1)
    model1.build()
    model1.train(n_epochs=40)
    model1.write_process('model1.csv')
    
    tf.reset_default_graph()
    model2 = ConvNet(cnn_layer_num=2, dnn_layer_num=1)
    model2.build()
    model2.train(n_epochs=40)
    model2.write_process('model2.csv')
    
    tf.reset_default_graph()
    model3 = ConvNet(cnn_layer_num=4, dnn_layer_num=1)
    model3.build()
    model3.train(n_epochs=40)
    model3.write_process('model3.csv')
    
    plot_result()
    
