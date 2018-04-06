# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 20:23:25 2018
@author: zhewei
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 13:58:27 2018
@author: zhewei
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 10:40:59 2018
@author: zhewei
"""

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
from sklearn.manifold import TSNE
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
SEED = 355

train_noise = np.random.normal(0 , 0.1 , [55000 , 784])
test_noise = np.random.normal(0 , 0.1 , [10000 , 784])

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
    def __init__(self, batch_size, store_path, store_name):
        self.lr = 0.001
        self.keep_prob = tf.constant(0.75)
        self.gstep = tf.Variable(0, dtype=tf.int32, 
                                trainable=False, name='global_step')
        self.n_classes = 10        
        self.batch_size = batch_size
        self.n_train = 55000
        self.n_test = 10000
        self.n_valid = 5000
        self.n_batch = int(self.n_train /self.batch_size)
        self.n_para = 0
        
        self.model_path = './' + self.create_save_path(store_path)
        self.store_name = store_name
        
        self.alpha_rec = []
        self.tr_loss_rec = []
        self.tr_acc_rec = []
        self.test_loss_rec = []
        self.test_acc_rec = []
        
    def create_save_path(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path
    
    def get_data(self):
        with tf.name_scope('data'):
            self.img = tf.placeholder(tf.float32, shape=[None, 784], name='img')
            self.label = tf.placeholder(tf.float32, shape=[None, 10], name='label')

    def inference(self):   
        print("Random initialize weights ... ") 
        img = tf.reshape(self.img, shape=[-1, 28, 28, 1])      
      
        
        print("Building 1-layer CNN  ...")
        self.conv1 = conv_relu(inputs=img, filters=10, k_size=5, scope_name='conv1')
        self.pool1 = maxpool(self.conv1, k_size=2, stride=2, scope_name='pool1')   
        
        feature_dim = self.pool1.shape[1] * self.pool1.shape[2] * self.pool1.shape[3]
        cnn_out = tf.reshape(self.pool1, [-1, feature_dim])  
        
        print("Building 4-Layer DNN ...")            
        self.fc1 = fully_connected(inputs=cnn_out, out_dim=10, scope_name='fc1')    
        self.fc2 = fully_connected(inputs=tf.nn.relu(self.fc1), out_dim=10, scope_name='fc2')                       
        self.fc3 = fully_connected(inputs=tf.nn.relu(self.fc2), out_dim=10, scope_name='fc3')                   
        self.fc4 = fully_connected(inputs=tf.nn.relu(self.fc3), out_dim=10, scope_name='fc4')
        self.logits = fully_connected(tf.nn.relu(self.fc4), self.n_classes, 'logits')
        
    
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
        '''
        writer = tf.summary.FileWriter(os.path.join('./graphs/',str(self.cnn_layer_num)+ str("_") +
                                                    str(self.dnn_layer_num)), tf.get_default_graph())
        '''
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.n_para = np.sum([np.prod(v.get_shape().as_list()) \
                                   for v in tf.trainable_variables()])
            print("Total Trainable parameters: {}".format(self.n_para))
            
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
    
                loss, acc = sess.run([self.loss, self.accuracy],
                                     feed_dict={self.img: mnist.test.images,
                                                self.label: mnist.test.labels})
                step += 1
                #writer.add_summary(summaries, global_step=step)    
                self.tr_acc_rec.append(tr_acc)
                self.tr_loss_rec.append(tr_loss)
                self.test_acc_rec.append(acc)
                self.test_loss_rec.append(loss)
                
                print('Training Loss : {0:.04f}'.format(tr_loss))
                print('Training Accuracy : {0:.04f} '.format(tr_acc))
                print('Testing Loss : {0:.04f}'.format(loss))
                print('Testing Accuracy : {0:.04f} '.format(acc), end='')
                print(', Took: {0:.02f} seconds'.format(time.time() - start_time))
            saver.save(sess, os.path.join(self.model_path, self.store_name))  


            tr_loss, tr_acc = sess.run([self.loss, self.accuracy],
                            feed_dict={self.img: mnist.train.images,
                                    self.label: mnist.train.labels})
            
            tr_noise_loss, tr_noise_acc = sess.run([self.loss, self.accuracy],
                                     feed_dict={self.img: mnist.train.images + train_noise,
                                                self.label: mnist.train.labels})
            
            te_loss, te_acc = sess.run([self.loss, self.accuracy],
                            feed_dict={self.img: mnist.test.images,
                                    self.label: mnist.test.labels})


            sensitive = abs(tr_noise_loss - tr_loss) / (np.sum(np.square(train_noise))/55000)



        return(tr_loss , tr_acc , te_loss , te_acc , sensitive)
          
    # run_acc_loss will call interpolate_parameters to calculate weighted sum 
    # theta = (1-alpha) * theta1 + alpha * theta2   
    # then call function assign_weights to assign new weights
    def run_acc_loss(self, model1, model2):
        alpha = np.linspace(-1, 2, 30)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for a in alpha:
                values_arr = self.interpolate_parameters(model1, model2, a)
                self.assign_weights(sess, values_arr)
                
                tr_loss, tr_acc = sess.run([self.loss, self.accuracy],
                                     feed_dict={self.img: mnist.train.images,
                                                self.label: mnist.train.labels})
                loss, acc = sess.run([self.loss, self.accuracy],
                                     feed_dict={self.img: mnist.test.images,
                                                self.label: mnist.test.labels})
                
                self.alpha_rec.append(a)
                self.tr_acc_rec.append(tr_acc)
                self.tr_loss_rec.append(tr_loss)
                self.test_acc_rec.append(acc)
                self.test_loss_rec.append(loss)
                print("alpha: {}".format(a))
                print("Loss: {}".format(tr_loss))
                print("Acc: {}".format(tr_acc))
                   
    def assign_weights(self, sess, values_arr):
        variables_names = [v.name for v in tf.trainable_variables()]
        for i in range(len(values_arr)):
             w = [w for w in tf.trainable_variables() if w.name == variables_names[i]]
             #print(w[0].name)
             sess.run(tf.assign(w[0], values_arr[i]))
        
    
    def interpolate_parameters(self, model1, model2, alpha):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            # restore model1's weights
            saver.restore(sess, os.path.join(self.model_path, model1)) 
            variables_names_1 = [v.name for v in tf.trainable_variables()]
            values_1 = sess.run(variables_names_1)
            
            # restore model2's weights
            saver.restore(sess, os.path.join(self.model_path, model2))
            variables_names_2 = [v.name for v in tf.trainable_variables()]
            values_2 = sess.run(variables_names_2)
            
            # interpolate
            values_1_arr = np.array(values_1) * alpha
            values_2_arr = np.array(values_2) * (1 - alpha)
            values_arr = values_1_arr + values_2_arr
            
            return values_arr  
     
    # output record to a file                         
    def write_process(self, filename):
        with open(filename, 'w') as fout:
            writer = csv.writer(fout, delimiter=',', lineterminator='\n')
            writer.writerow(["alpha","train_acc","train_loss","test_acc","test_loss"])
            for i in range(len(self.tr_acc_rec)):
                writer.writerow([str(self.alpha_rec[i]), 
                                     str(self.tr_acc_rec[i]),str(self.tr_loss_rec[i]),
                                     str(self.test_acc_rec[i]), str(self.test_loss_rec[i])])

                                  
        
def plot_result(filename):
    dframe = pd.read_csv(filename, sep=',', header=0)
    alpha = np.array(dframe['alpha'])
    tr_acc = np.array(dframe['train_acc'])
    tr_loss = np.array(dframe['train_loss'])
    acc = np.array(dframe['test_acc'])
    loss = np.array(dframe['test_loss'])
    
    # twin_plot
    print("plot loss/acc - alpha ...")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    p1 = ax.plot(alpha, tr_acc, color='red' , label = 'train_acc')
    p2 = ax.plot(alpha, acc, color='red', linestyle='--', label='test_acc')
    ax2 = ax.twinx()
    p3 = ax2.plot(alpha, tr_loss, color='blue', label = 'train_loss')
    p4 = ax2.plot(alpha, loss, color='blue', linestyle='--', label='test_loss')
    # get labels and show with legend
    p = p1 + p2 + p3 + p4
    labs = [l.get_label() for l in p]
    ax.legend(p, labs, loc=0)
    
    # set title and label
    ax.set_ylabel('Accuracy', color='red')
    ax.set_title("acc/loss - alpha")
    ax.tick_params('y', color='red')
    
    ax2.set_xlim([-1, 2])
    ax2.set_yscale("log")
    ax2.set_ylabel('Loss', color='blue')
    ax2.set_xlabel('alpha')
    ax2.tick_params('y', color='blue')
    
    plt.show()
    
if __name__ == '__main__':
    batch = [10 , 50 , 100 , 500 , 1000]
    sensitive = []
    tr_loss = []
    tr_acc = [] 
    te_loss = []
    te_acc = [] 

    tf.reset_default_graph() 
    model = ConvNet(batch_size=10, store_path='model', store_name='m1')
    model.build()
    tr_l , tr_a , te_l , te_a , sen = model.train(n_epochs=10)
    tr_loss.append(tr_l)
    tr_acc.append(tr_a)
    te_loss.append(te_l)
    te_acc.append(te_a)
    sensitive.append(sen)

    tf.reset_default_graph() 
    model = ConvNet(batch_size=50, store_path='model', store_name='m2')
    model.build()
    tr_l , tr_a , te_l , te_a , sen = model.train(n_epochs=20)
    tr_loss.append(tr_l)
    tr_acc.append(tr_a)
    te_loss.append(te_l)
    te_acc.append(te_a)
    sensitive.append(sen)

    tf.reset_default_graph() 
    model = ConvNet(batch_size=100, store_path='model', store_name='m3')
    model.build()
    tr_l , tr_a , te_l , te_a , sen = model.train(n_epochs=20)
    tr_loss.append(tr_l)
    tr_acc.append(tr_a)
    te_loss.append(te_l)
    te_acc.append(te_a)
    sensitive.append(sen)

    tf.reset_default_graph() 
    model = ConvNet(batch_size=500, store_path='model', store_name='m4')
    model.build()
    tr_l , tr_a , te_l , te_a , sen = model.train(n_epochs=30)
    tr_loss.append(tr_l)
    tr_acc.append(tr_a)
    te_loss.append(te_l)
    te_acc.append(te_a)
    sensitive.append(sen)

    tf.reset_default_graph() 
    model = ConvNet(batch_size=1000, store_path='model', store_name='m5')
    model.build()
    tr_l , tr_a , te_l , te_a , sen = model.train(n_epochs=40)
    tr_loss.append(tr_l)
    tr_acc.append(tr_a)
    te_loss.append(te_l)
    te_acc.append(te_a)
    sensitive.append(sen)


    fig, ax1  = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(batch , tr_loss , "b" , label = 'train')
    ax1.plot(batch , te_loss , "b--" ,  label = 'test' )
    ax2.plot(batch , sensitive , "r" , label = 'sensitive')
    ax1.set_xlabel('batch size')
    ax1.set_ylabel('loss')
    ax2.set_ylabel('sensitive')
    ax1.legend(loc=1) 
    ax2.legend(loc=2) 
    plt.show()
################################################################
    fig, ax3  = plt.subplots()
    ax4 = ax3.twinx()

    ax3.plot(batch , tr_acc , "b" , label = 'train')
    ax3.plot(batch , te_acc , "b--" ,  label = 'test' )
    ax4.plot(batch , sensitive , "r" , label = 'sensitive')
    ax3.set_xlabel('batch size')
    ax3.set_ylabel('accuracy')
    ax4.set_ylabel('sensitive')
    ax3.legend(loc=1) 
    ax4.legend(loc=2) 
    plt.show()


    """
    tf.reset_default_graph() 
    model = ConvNet(batch_size=50, store_path='model', store_name='m3')
    model.build()
    model.run_acc_loss('m1', 'm2')
    model.write_process('loss_acc.csv')
    
    plot_result('loss_acc.csv')"""
    