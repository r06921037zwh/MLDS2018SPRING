# -*- coding: utf-8 -*-
"""
Created on Tue May 29 00:48:45 2018

@author: BananaWang CGAN(Conditional GAN)
"""

import numpy as np
import sys
import os
from scipy import misc
import tensorflow as tf
import csv
import pandas as pd
import re
import matplotlib.pyplot as plt
import itertools



img_size = 96
img_dir = 'D:/MLDS/HW3/faces'
tag_file = 'D:/MLDS/HW3/tags_clean.csv'
noise_size = 100 # size_z

def read_imgs(img_dir): 
    
    true_imgs_dic = {}
    true_imgs = []
    
    for root, dirs, filenames in os.walk(img_dir):
        for f in filenames:
            im = misc.imread(os.path.join(img_dir,f))
            true_imgs_dic[int(f.replace('.jpg',''))] = im  
    
    for i in range(len(true_imgs_dic)): # reorder the files numerically 
        true_imgs.append((misc.imresize(true_imgs_dic[i], [64, 64, 3]).astype(np.float)/127.5) -1)
        
    true_imgs_dic = None # erase from the memory
    # reshape the image to 64*64
    return true_imgs


def build_dict():
    hair_dict = {'orange hair':12, 'white hair':1, 'aqua hair':2, 'gray hair':3,
                 'green hair':4, 'red hair':5, 'purple hair':6, 'pink hair':7,
                 'blue hair':8, 'black hair':9, 'brown hair':10, 'blonde hair':11}
    
    eye_dict = {'gray eyes':11, 'black eyes':1, 'orange eyes':2,
                'pink eyes':3, 'yellow eyes':4, 'aqua eyes':5, 'purple eyes':6,
                'green eyes':7, 'brown eyes':8, 'red eyes':9, 'blue eyes':10}
            
    return hair_dict, eye_dict
            

def read_test(test_file, hair_dict, eye_dict):
    f = open(test_file, 'r')
    test_data = []
      
    for line in f:
       temp = line.split(',')[1].split()
       print(temp)
       test_data.append((temp[0]+' '+temp[1], temp[2]+' '+temp[3]))
    f.close()
    
    test_data_array = np.zeros(shape=(len(test_data), 2))
    for i in range(len(test_data)):
        test_data_array[i,0] = hair_dict[test_data[i][0]]
        test_data_array[i,1] = eye_dict[test_data[i][1]]
          
    return test_data_array # now each image with only 2 numbers


def parse_tags(tag_file, hair_dict, eye_dict):
    tags = pd.read_csv(tag_file, header=None)
    img_attributes = np.zeros(shape=(tags.shape[0],2)) # [0] hair 
    tags = tags.iloc[:].values
    for i in range(len(tags)):
        temp = tags[i][1].split("\t")
        for j in temp:
            j = re.sub("[0-9]+","",j)
            j = re.sub(':',"",j)
            
            if j in hair_dict:
                img_attributes[i,0] = hair_dict[j]
            if j in eye_dict:
                img_attributes[i,1] = eye_dict[j]
    
    return img_attributes 



def pre_select_training_data(img_attributes, true_imgs):
    refined_img = []
    index = []
    for i in range(len(img_attributes)):
        if img_attributes[i,0]!=0 or img_attributes[i,1]!=0:
            refined_img.append(img_attributes[i,:].astype(int))
            index.append(i)
    
    refined_img_attributes = np.zeros(shape=(len(index),2))
    refined_true_imgs = np.zeros(shape=(len(index), 64, 64, 3))
    for i in range(len(index)):
        refined_img_attributes[i,:] = img_attributes[index[i],:]
        refined_true_imgs[i] = true_imgs[index[i]]
            
    print(len(refined_true_imgs))
    print(len(refined_img_attributes))
      
    
    return refined_true_imgs, refined_img_attributes

#refined_true_imgs, refined_img_attributes = pre_select_training_data(img_attributes, true_imgs)



class Conditional_GAN(object):
    
    def __init__(self):

         self.embedding_size = 20
         self.img_w = 64
         self.img_h = 64
         self.batch_size = 128 
         self.data_size = 17395#33431
         self.kernel_size = 5
         self.stride_size = 2
         self.random_noise_dim = 113
         self.lr = 0.0002
         self.tag_dim = 2
         self.n_epochs = 160
         self.G_epochs = 1
         self.D_epochs = 5
         self.model_path = 'D:/MLDS/HW3/GANmodel-8'
         self.testing_requirements = np.array([[8,10],[8,10],[8,10],[8,10],[8,10],[8,7],[8,7],[8,7],[8,7],[8,7],[8,9],[8,9],[8,9],[8,9],[8,9],[4,10],[4,10],[4,10],[4,10],[4,10],[4,9],[4,9],[4,9],[4,9],[4,9]]) #Commemorate 6/4
         self.test_use_random_batch_z = np.random.normal(-0.4, 0.4, (25, self.random_noise_dim))

         
    def generator(self, z, tag_vec):
        
        with tf.variable_scope("g_net") as scope:             
            #scope.reuse_variables()           
            # +1 for the un-tagged data   
            hair_vec = tf.nn.embedding_lookup(self.hair_emb, tag_vec[:,0])
            eyes_vec = tf.nn.embedding_lookup(self.eyes_emb, tag_vec[:,1])
            
            #tag_embedded_vec = tf.contrib.layers.fully_connected(tag_vec, self.embedding_size, activation_fn = None)            
            g_input = tf.concat([hair_vec, eyes_vec, z], axis=1) #g_input(128,201)
            #tf.get_variable_scope().reuse_variables()
            g1 = tf.layers.dense(g_input, 4*4*256, kernel_initializer=tf.random_normal_initializer(stddev=0.01), reuse = tf.AUTO_REUSE)
            g1 = tf.layers.batch_normalization(g1, training=True, reuse=tf.AUTO_REUSE)
            g1 = tf.reshape(g1, [-1, 4, 4, 256])
            g1 = tf.nn.relu(g1)
        
            gc1 = tf.contrib.layers.convolution2d_transpose(g1, 128, [self.kernel_size, self.kernel_size], [self.stride_size, self.stride_size], padding='same', weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
            gc1 = tf.layers.batch_normalization(gc1, training=True)
            gc1 = tf.nn.relu(gc1)
            
            gc2 = tf.contrib.layers.convolution2d_transpose(gc1, 64, [self.kernel_size, self.kernel_size], [self.stride_size, self.stride_size], padding='same', weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
            gc2 = tf.layers.batch_normalization(gc2, training=True)
            gc2 = tf.nn.relu(gc2)
            
            gc3 = tf.contrib.layers.convolution2d_transpose(gc2, 32, [self.kernel_size, self.kernel_size], [self.stride_size, self.stride_size], padding='same', weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
            gc3 = tf.layers.batch_normalization(gc3, training=True)
            gc3 = tf.nn.relu(gc3)
                
            gc4 = tf.contrib.layers.convolution2d_transpose(gc3, 3, [self.kernel_size, self.kernel_size], [self.stride_size, self.stride_size], padding='same', weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
            print(gc4)
            g_out = tf.nn.tanh(gc4) # the generated image (64*64s)
            
        return g_out
    
    def discriminator(self,input_img, input_tag_vec, reuse):  #self.input_tag_vec (batchs_size, 2)
        
        print('---------------------------------111---------------------------------------')

        with tf.variable_scope("g_net") as scope:
            scope.reuse_variables()
            hair_vec = tf.nn.embedding_lookup(self.hair_emb, input_tag_vec[:,0])
            eyes_vec = tf.nn.embedding_lookup(self.eyes_emb, input_tag_vec[:,1])
            #tag_embedded_vec = tf.contrib.layers.fully_connected(input_tag_vec, self.embedding_size, activation_fn = None)            
            tag_embedded_vec = tf.concat([hair_vec, eyes_vec], axis = 1)  
        
        with tf.variable_scope("d_net") as scope:             
            if reuse == True:
                scope.reuse_variables()
            
            dc1 = tf.contrib.layers.conv2d(input_img, 32, [self.kernel_size, self.kernel_size], [self.stride_size, self.stride_size],padding='same', weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
        		#dc1 = tf.layers.batch_normalization(dc1, training=True)
            dc1 = tf.nn.leaky_relu(dc1)
            print('---------------------------------222---------------------------------------')

            dc2 = tf.contrib.layers.conv2d(dc1, 64, [self.kernel_size, self.kernel_size], [self.stride_size, self.stride_size],padding='same', weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
            dc2 = tf.layers.batch_normalization(dc2, training=True)
            dc2 = tf.nn.leaky_relu(dc2) # (128, 16, 16, 128)
            
            dc3 = tf.contrib.layers.conv2d(dc2, 128, [self.kernel_size, self.kernel_size], [self.stride_size, self.stride_size],padding='same', weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
            dc3 = tf.layers.batch_normalization(dc3, training=True)
            dc3 = tf.nn.leaky_relu(dc3) # (128, 8, 8, 256)
       
            dc4 = tf.contrib.layers.conv2d(dc3, 256, [self.kernel_size, self.kernel_size], [self.stride_size, self.stride_size],padding='same', weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
            dc4 = tf.layers.batch_normalization(dc4, training=True)
            dc4 = tf.nn.leaky_relu(dc4) # (128, 4, 4, 512)
            
            #tag_vec = tf.contrib.layers.fully_connected(the_tag, self.embedding_size, activation_fn = None)
            tag_vec = tf.expand_dims(tf.expand_dims(tag_embedded_vec, 1), 2)
            tag_vec = tf.tile(tag_vec, [1, 4, 4, 1])
            
            dc5_input = tf.concat([dc4, tag_vec], axis=-1)
            
            dc5 = tf.contrib.layers.conv2d(dc5_input, 256, [1,1], [1,1], padding='same', weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
            dc5 = tf.layers.batch_normalization(dc5, training=True)
            dc5 = tf.nn.leaky_relu(dc5)            
           
            dc6 = tf.contrib.layers.conv2d(dc5, 1, [4, 4], [1, 1], padding='valid', weights_initializer=tf.random_normal_initializer(stddev=0.02), activation_fn=None)
            d_out = tf.squeeze(dc6, [1, 2, 3])

            #dc5_flat = tf.layers.Flatten()(dc5)
            #print(dc5_flat)            
            #d_out = tf.contrib.layers.fully_connected(dc5_flat, 1, activation_fn = tf.nn.sigmoid)
            #print(d_out)  
        return d_out
        
    """        
    def embedding(self, the_tag, reuse):
        with tf.variable_scope("emb") as scope:             
            if reuse == True:
                scope.reuse_variables()
            
            self.embedded_tags = tf.contrib.layers.fully_connected(the_tag, self.embedding_size, activation_fn = None) # tag_id should from placeholder
        return self.embedded_tags
    """        
    def build_model(self):
        tf.reset_default_graph()
        
        self.hair_emb = tf.Variable(tf.random_uniform([12+1, self.embedding_size], -1, 1), name='g_net/hair_emb')         
        self.eyes_emb = tf.Variable(tf.random_uniform([11+1, self.embedding_size], -1, 1), name='g_net/eyes_emb')
        
        self.tag = tf.placeholder(tf.int32, [None, self.tag_dim]) 
        # None will make the input size more flexible
        self.wrong_tags = tf.placeholder(tf.int32, [None, self.tag_dim]) 
        # tag(128,2)
        """
        self.embedded_tags = self.embedding(self.tag, reuse = False) # tag_id should from placeholder
        self.embedded_wrong_tags = self.embedding(self.wrong_tags, reuse = True)
        """
        # embedded_tags (128,100)
            
        self.z = tf.placeholder(tf.float32, [None, self.random_noise_dim]) #self.z (128,128)
        self.G_img = self.generator(self.z, self.tag)    
        self.input_img = tf.placeholder(tf.float32, [None,64,64,3])
        # when feed in, just feed the same labels, the random shuffles will be done inside the train()
        self.D_TT = self.discriminator(self.input_img, self.tag, reuse = False)
        self.D_TF = self.discriminator(self.input_img, self.wrong_tags, reuse = True)
        self.D_FT = self.discriminator(self.G_img, self.tag, reuse = True)
        self.D_FF = self.discriminator(self.G_img, self.wrong_tags, reuse = True)

        
        t_vars = tf.trainable_variables()
        self.d_var = [var for var in t_vars if 'd_net' in var.name]
        self.g_var = [var for var in t_vars if 'g_net' in var.name]
            
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_FT, labels=tf.ones_like(self.D_FT))) 
    
        self.D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_TT, labels=tf.ones_like(self.D_TT)))+ \
        				(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_TF, labels=tf.zeros_like(self.D_TF)))+ \
    					   tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_FT, labels=tf.zeros_like(self.D_FT)))+ \
    					   tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_FF, labels=tf.zeros_like(self.D_FF))) ) / 3  		

       
        self.opti_D = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5).minimize(self.D_loss, var_list=self.d_var)
        self.opti_G = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5).minimize(self.G_loss, var_list=self.g_var)
        
    
    
    def train(self, true_imgs, img_attributes):
        
        saver = tf.train.Saver(max_to_keep=300)
        with tf.Session() as sess:   
            tf.global_variables_initializer().run()
            for epoch in range(self.n_epochs): # parse the data [batch_size] here!! (start, end)
                print('-------------------------'+ str(epoch) + '--------------------------')
                for start, end in zip(range(0, self.data_size, self.batch_size), range(self.batch_size, self.data_size, self.batch_size)):

                    batch_images_real = np.array(refined_true_imgs[start:end]) #input 
                    tags_correct = np.array(refined_img_attributes[start:end,:])
                    batch_z = np.random.normal(0, 1, size=[self.batch_size, self.random_noise_dim]).astype(np.float32)
                    tags_wrong = np.random.permutation(tags_correct)
                    #print(tags_wrong.shape)
                    #print(tags_correct.shape)
                # train Generator 
               
                    for G_epoch in range(self.G_epochs):
                        sess.run(self.opti_G, feed_dict={self.tag: tags_correct,
                                                         self.z: batch_z})
                                        
                    for D_epoch in range(self.D_epochs):
                        sess.run(self.opti_D, feed_dict={self.input_img: batch_images_real, 
                                                         self.tag: tags_correct,
                                                         self.wrong_tags: tags_wrong,
                                                         self.z: batch_z})
                   
                
                if epoch % 5 == 0:
                    self.show_img(sess, self.testing_requirements, epoch, False)
                    
                if (epoch+1) % 20 == 1:                    
                    saver.save(sess, os.path.join(self.model_path, "model"), global_step=epoch)
                   
               # train discriminator
        
        
        
          
    def test(self, saved_model_path, test_data, test_flag): # restore the session and saved file here
           
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, saved_model_path)
            self.show_img(sess, test_data, 0, test_flag)
                
                 
            
    
    def show_img(self, sess, test_data_requirements, epoch, test_flag): # test_data_requirements [4,2]
        if test_flag == True:
            self.test_use_random_batch_z = np.random.normal(-0.4, 0.4, (25, self.random_noise_dim))
            print('regenerate initial seeds')
            np.save('./best_random_25s.npy', self.test_use_random_batch_z)
        
        generated_imgs = sess.run(self.G_img, {self.z: self.test_use_random_batch_z, self.tag: self.testing_requirements}) # fix the random data
    
        size_figure_grid = 5
        fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(8, 8))
        for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
            ax[i, j].get_xaxis().set_visible(False)
            ax[i, j].get_yaxis().set_visible(False)
    
        for k in range(size_figure_grid*size_figure_grid):
            i = k // size_figure_grid
            j = k % size_figure_grid
            ax[i, j].cla()
            ax[i, j].imshow(self.image_recover(generated_imgs[k]))  
        #label = 'Epoch {0}'.format(self.n_epochs)
        plt.savefig('./samples/cgan.png')   # here needs to be modified 
        plt.show()
        
    
    
    def image_recover(self, G_img_tensor): # pixel range [-1, 1] --> [0, 255]
        
        G_img_tensor = G_img_tensor.copy().astype(float)
        G_min = G_img_tensor.min()
        #print(G_min)
        img_range = G_img_tensor.max() - G_min
        #print(img_range)
        G_img_tensor = ((G_img_tensor - G_min) / img_range) * 255
        
        return G_img_tensor.astype(np.uint8) 



# real main function 
"""
def main():
    hair_dict, eye_dict = build_dict()
    img_attributes = parse_tags(tag_file, hair_dict, eye_dict)
    refined_true_imgs, refined_img_attributes = pre_select_training_data(img_attributes, true_imgs)
    test_file = sys.argv[1]
    test_data = read_test(test_file, hair_dict, eye_dict)
    true_imgs = read_imgs(img_dir) # read the training data from directory
    cgan = Conditional_GAN()
    cgan.build_model()
    cgan.train(refined_true_imgs, refined_img_attributes)
"""







"""
def main():
    hair_dict, eye_dict = build_dict()
    test_file = sys.argv[1]
    test_data = read_test(test_file, hair_dict, eye_dict)
    cgan = Conditional_GAN()
    cgan.build_model()
    cgan.test('D:/MLDS/HW3/GANmodel-8/model-60', testing_requirements, True)   #test_data
    

def main():
    
"""

def main():            
      
    hair_dict, eye_dict = build_dict()
    test_file = sys.argv[1]
    test_data = read_test(test_file, hair_dict, eye_dict)
    cgan = Conditional_GAN()
    cgan.build_model()
    cgan.test('./cgan_model', test_data, False)   #test_data

      
        
   # def visualize():
        
        
        
if __name__ == "__main__":     
   main()     
        
   # def save_model():
            
