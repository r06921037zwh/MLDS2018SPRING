# -*- coding: utf-8 -*-
"""
Created on Fri May 25 23:33:27 2018

@author: zhewei
"""

import tensorflow as tf 
import os 
import itertools
import numpy as np
import matplotlib.pyplot as plt
from utils import Dataset, read_imgs

imgs_dir = 'faces/*.jpg'
logdir = 'tf_logs/'

class DCGAN(object):
    def __init__(self, train_set, model_dir='dcgan_model', model_name='dcgan.ckpt'):
        self.train_set = train_set
        self.model_path = model_dir
        self.model_name = model_name
        self.batch_size = 100
        self.training_epochs = 300
        self.clip = 0.01
        self.noise_dim = 100
        self.out_channel_dim = 3
        # create graph
        self.create_graph()
          
    
    def leaky_ReLU_activation(self, x, alpha=0.2):
        """Leaky ReLU activation function"""
        return tf.maximum(alpha * x, x)
    
    def dropout(self, x, keep_prob=1.0):
        """Dropout function"""
        return tf.nn.dropout(x, keep_prob)
    
    def d_conv(self, x, filters, kernel_size, strides, padding='same', alpha=0.2, keep_prob=1.0, train=True):
        """
        Discriminant layer architecture
        Creating a convolution, applying batch normalization, leaky rely activation and dropout
        """
        x = tf.layers.conv2d(x, filters, kernel_size, strides, padding,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())
        x = tf.layers.batch_normalization(x, training=train)
        x = self.leaky_ReLU_activation(x, alpha)
        x = self.dropout(x, keep_prob)
        return x

    def g_reshaping(self, x, shape, alpha=0.2, keep_prob=1.0, train=True):
        """
        Generator layer architecture
        Reshaping layer, applying batch normalization, leaky rely activation and dropout
        """
        x = tf.reshape(x, shape)
        x = tf.layers.batch_normalization(x, training=train)
        x = self.leaky_ReLU_activation(x, alpha)
        x = self.dropout(x, keep_prob)
        return x

    def g_conv_transpose(self, x, filters, kernel_size, strides, padding='same', alpha=0.2, keep_prob=1.0, train=True):
        """
        Generator layer architecture
        Transposing convolution to a new size, applying batch normalization, leaky rely activation and dropout
        """
        x = tf.layers.conv2d_transpose(x, filters, kernel_size, strides, padding)
        x = tf.layers.batch_normalization(x, training=train)
        x = self.leaky_ReLU_activation(x, alpha)
        x = self.dropout(x, keep_prob)
        return x
    
    def discriminator(self, images, labels=None):
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            # Input layer is 64x64x3 --> concatenating images 
            if labels != None:
                x = tf.concat([images, labels], 3)
            
            # d_conv --> expected size is 32x32x32
            x = self.d_conv(images, filters=32, kernel_size=4, strides=2, padding='same',
                            alpha=0.2, keep_prob=1.0)

            # d_conv --> expected size is 16x16x64
            x = self.d_conv(x, filters=64, kernel_size=4, strides=2, padding='same',
                            alpha=0.2, keep_prob=1.0)

            # d_conv --> expected size is 8x8x128
            x = self.d_conv(x, filters=128, kernel_size=4, strides=2, padding='same',
                            alpha=0.2, keep_prob=1.0)

            # d_conv --> expected size is 8x8x256
            x = self.d_conv(x, filters=256, kernel_size=4, strides=1, padding='same',
                            alpha=0.2, keep_prob=1.0)
            
            # Flattening to a single layer --> expected size is 4096
            x = tf.reshape(x, (-1, 8 * 8 * 256))

            # Calculating logits and sigmoids
            logits = tf.layers.dense(x, 1)
            sigmoids = tf.sigmoid(logits)

            return sigmoids, logits

    def generator(self, z, out_channel_dim, is_train=True):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            # First fully connected layer
            x = tf.layers.dense(z, 8 * 8 * 512)

            # Reshape it -> 8x8x512
            x = self.g_reshaping(x, shape=(-1, 8, 8, 512), alpha=0.2, keep_prob=1.0, train=is_train)

            # g_conv_transpose --> 16x16x256 now
            x = self.g_conv_transpose(x, filters=256, kernel_size=4, strides=2, padding='same',
                              alpha=0.2, keep_prob=1.0, train=is_train)

            # g_conv_transpose --> 32x32x128 now
            x = self.g_conv_transpose(x, filters=128, kernel_size=4, strides=2, padding='same',
                              alpha=0.2, keep_prob=1.0, train=is_train)
            
            # g_conv_transpose --> 64x64x32 now
            x = self.g_conv_transpose(x, filters=64, kernel_size=4, strides=2, padding='same',
                              alpha=0.2, keep_prob=1.0, train=is_train)
            
            # Calculating logits and Output layer --> 64x64x3 now
            logits = tf.layers.conv2d_transpose(x, filters=out_channel_dim, kernel_size=4, strides=1, padding='same')
            output = tf.tanh(logits)

            return output

    def optimize_op(self, cgan=False):
        self.isTrain = tf.placeholder(dtype=tf.bool)
        self.real_x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
        self.random_x = tf.placeholder(tf.float32, shape=[None, self.noise_dim])
        if cgan:
            self.random_x = tf.placeholder(tf.float32, shape=[None, self.noise_dim + self.classes])
        self.random_y = self.generator(self.random_x, self.out_channel_dim)

        
        # networks : discriminator
        D_real_sigmoid, D_real_logits = self.discriminator(self.real_x)
        D_fake_sigmoid, D_fake_logits = self.discriminator(self.random_y)
        
        # loss for each network
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_logits) * 0.9))
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))
        self.D_loss = D_loss_real + D_loss_fake
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))            
        
        
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]
        
        self.D_solver = tf.train.AdamOptimizer(2e-4, 0.5).minimize(self.D_loss, var_list=d_vars)
        self.G_solver = tf.train.AdamOptimizer(2e-4, 0.5).minimize(self.G_loss, var_list=g_vars)   
   
    def create_graph(self):
        print("Building Graph ...")
        self.optimize_op()
        self.summary()
        
    def summary(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('G_loss', self.G_loss)
            self.summary_op = tf.summary.merge_all()
            
    def train(self):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        self.saver = tf.train.Saver()
        self.file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
        
        with tf.Session() as sess:
            # if having pretrained model, use it; or, train a new one    
            ckpt = tf.train.get_checkpoint_state(os.path.join(self.model_path, '1000ePoch'))  
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print('Reloading model parameters..')
                self.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('Created new model parameters..')
                sess.run(tf.global_variables_initializer())
                           
            print("Model Training ... ")
            step = 0
            for epoch in range(self.training_epochs):
                print("===== epoch {} =====".format(epoch + 1))
                                
                for batch_images in self.train_set.get_batches(self.batch_size):                  
                    step += 1
                    # update discriminator
                    real_batch_x = batch_images
                    random_batch_x = np.random.uniform(-1, 1, (len(batch_images), self.noise_dim))
                    _ , D_loss_ = sess.run([self.D_solver, self.D_loss], 
                                          feed_dict={self.real_x: real_batch_x,
                                                     self.random_x: random_batch_x})
                    # update generator
                    if step % 1 == 0:
                        random_batch_x = np.random.uniform(-1, 1, (len(batch_images), self.noise_dim))
                        _, G_loss_ , summaries = sess.run([self.G_solver, self.G_loss, self.summary_op], 
                                                          feed_dict={self.random_x: random_batch_x})
                        self.file_writer.add_summary(summaries, global_step=step)
                           
                # write summary using Filewriter to a logfile
                print('epoch {} :  D_loss {:.6f} , G_loss {:.6f}'.format(epoch + 1, D_loss_, G_loss_))
                
                # print images every 5 epochs
                if epoch % 1 == 0: 
                    self.show_result(sess, (epoch + 1))
                    self.save_imgs(sess, seed=0, path='out' ,name=('gan-original' + str(epoch) + '.png'))
                # save model every 20 epochs
                if epoch % 20 == 0:
                    checkpoint_dir = os.path.join(self.model_path, str(epoch + 1)+'ePoch')
                    self.checkpoint_path = os.path.join(checkpoint_dir, self.model_name)
                    print("Saving model named {}epoch  ...".format(epoch + 1))
                    self.saver.save(sess, self.checkpoint_path)
                    print("Model saved ... ")
                
        self.file_writer.close()
    
    def rescale_images(self, image_array):
        """
        Scaling images in the range 0-255
        """
        new_array = image_array.copy().astype(float)
        min_value = new_array.min()
        range_value = new_array.max() - min_value
        new_array = ((new_array - min_value) / range_value) * 255
        return new_array.astype(np.uint8)
    
    
    def show_result(self, sess, num_epoch):
        random_batch_x = np.random.uniform(-1, 1, (25, self.noise_dim))
        generated_imgs = sess.run(self.random_y, {self.random_x: random_batch_x})
    
        size_figure_grid = 5
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(8, 8))
        for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
    
        for k in range(size_figure_grid*size_figure_grid):
            i = k // size_figure_grid
            j = k % size_figure_grid
            ax[i, j].cla()
            ax[i, j].imshow(self.rescale_images(generated_imgs[k]))  
        label = 'Epoch {0}'.format(num_epoch)
        fig.text(0.5, 0.04, label, ha='center')
        plt.show()
        
    
    def save_imgs(self, sess, seed, path, name):
        if not os.path.exists(path):
            os.makedirs(path)
            
        r, c = 5, 5
        np.random.seed(seed)
        random_batch_x = np.random.uniform(-1, 1, (r * c, self.noise_dim))
        gen_imgs = sess.run(self.random_y, {self.random_x: random_batch_x})
        
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(self.rescale_images(gen_imgs[cnt, :,:,:]))
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(path, name))
        plt.close()
    
    def generate_with_model(self, model_name, save_path, save_name, seed):
        self.saver = tf.train.Saver()
        
        with tf.Session() as sess:
            # if having pretrained model, use it; or, train a new one    
            ckpt = tf.train.get_checkpoint_state(os.path.join(self.model_path, model_name))     
            print('Reloading model parameters..')
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            
            # generate_images
            print('Generating Faces ...')
            self.save_imgs(sess, seed, path=save_path ,name=save_name)
            
if __name__ == '__main__':
    tf.reset_default_graph()
    #imgs = read_imgs(imgs_dir)
    #dataset = Dataset(imgs)
    dataset = []
    dcgan = DCGAN(dataset, model_dir=os.path.join('dcgan_model'), model_name='dcgan.ckpt')
    #dcgan.train()
    dcgan.generate_with_model(model_name='300ePoch', save_path='samples', save_name='gan.png', seed=56)            
