# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 11:01:27 2018

@author: BananaWang
"""

import os
import sys
import numpy as np
import json
import re
import tensorflow as tf
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
import time 
import random
import math 
  
def read_training_data():
    train_X = np.zeros((1450,80,4096),dtype=np.float32)
    label_X = []
    count = 0
    indir = 'C:/Users/RUMI/Desktop/MLDS_hw2_1_data/training_data/feat' 
    with open("C:/Users/RUMI/Desktop/MLDS_hw2_1_data/training_label.json") as jsf:
        json_data = json.load(jsf)
        for root, dirs, filenames in os.walk(indir):
            for f in filenames:
                train_X[count,:] = np.load(os.path.join(indir,f))
                count+=1
                for i in json_data:                    
                    if i['id'] == f.replace(".npy",""):
                        label_X.append(i["caption"])
                        
    return (train_X, label_X, json_data) # this matches the order 


# build the dictionary and inverse dictionary

def build_dic(json_data, min_freq):
    #min_freq =3
    print('Creating the dictionary with min word count =  %d' % (min_freq))

    
    
    dictionary = {'PAD':[0,min_freq+1],'BOS':[1,min_freq+1], 'EOS':[2,min_freq+1], 'UKN':[3, min_freq+1]}
    if len(dictionary)>3: # 0 for unseen or rare words; 1 for start_token; 2 for end_token; 3 for padding to same len
        order = len(dictionary)
    else:
        order = 4
    for item in json_data:
        for line in item['caption']:
            line = re.sub(r"[\.,]","",line)
            for word in line.lower().split():
                  if word in dictionary:       
                      dictionary[word][1] = dictionary[word][1]+1
                  else:
                      dictionary[word] = [order, 1]
                      order = order +1  
    
    
    vocab = [w for w in dictionary if dictionary[w][1] >= min_freq]
    print(len(vocab))

    return vocab



def read_testing_json():
    test_features = np.zeros((100,80,4096),dtype=np.float32)
    test_labels = []
    testing_order = []
    count = 0 
    #with open("D:/Downloads/MLDS_hw2_1_data/testing_label.json") as jsf:
    #    json_testing_data = json.load(jsf)
    indir = sys.argv[1]
    
    for root, dirs, filenames in os.walk(indir):
            for f in filenames:
                test_features[count,:] = np.load(os.path.join(indir,f))
                count+=1
                testing_order.append(f.replace(".npy",""))
        
    return (test_features, testing_order)
    # test labels are for validation (calculate the bleu score)


def data_preproc(json_data):
# The dataset is too huge, thus encoding(embedding is done for only the data to be processed)
# build the batch generator
    total_training_samples = 0 
    repetition_list = []
    for item in json_data:
        a = len(item['caption'])
        total_training_samples += a
        repetition_list.append(a)
    print('====================================')
    print(total_training_samples)        
    print('====================================')
    return (total_training_samples, repetition_list)


def train_data_processor(json_data, wordtoindex, total_training_samples):
    # This function make sentences to indecis and also pad BOS EOS and UKN
    # this function returns the data in an array
    training_data_vec = [] # 20 is the human defined sentence length
    for item in json_data:
        for sentence in item['caption']:
            sentence_length = 0
            sentence = re.sub(r"[\.,]","",sentence)
            sentence_vec = [wordtoindex['BOS']]
            
            for word in sentence.lower().split():
                sentence_length+=1
                if word in wordtoindex:
                    sentence_vec.append(wordtoindex[word])
                else:
                    sentence_vec.append(wordtoindex['UKN'])
                if sentence_length is 19:
                    break
            sentence_vec.append(wordtoindex['EOS'])
            training_data_vec.append(sentence_vec)
            
    return training_data_vec


def mapping_training_data_dic():
    index = 0
    mapping_dict = {} 
    
    for i in range(len(repetition_list)):
        for j in range(index, index + repetition_list[i]):
            mapping_dict[j] = i
        index+=repetition_list[i]
    return mapping_dict

#########################################
# Global Variables
data_size = 1450   
n_epochs = 1000 
lr_rate = 0.00001 
 
dim_image = 4096
lstm_size= 512

n_video_lstm_step = 80
n_caption_lstm_step = 20
n_frame_step = 80
batch_size = 100
model_path = './my_models/model_100nn'
#########################################


# build the embedding mapping funcitons(convert the label_X to usable labels(same length and one hot encoding))
"""train_X, label_X, json_data= read_training_data()
vocab = build_dic(json_data, 1)

wordtoindex = {} # build embedding matrix 
for i in range(len(vocab)):
    wordtoindex[vocab[i]]= i
   
indextoword = {} # build inverse-embedding matrix 
for word in wordtoindex:
    indextoword[wordtoindex[word]] = word"""

indextoword = np.load('my_dict.npy').item()
    
test_features, testing_order = read_testing_json()
"""
(total_training_samples, repetition_list) = data_preproc(json_data)
training_data_vec = train_data_processor(json_data, wordtoindex, total_training_samples)"""



class Video_Caption_Generator():
    def __init__(self, dim_image, n_words, lstm_size, batch_size, n_lstm_steps, n_video_lstm_step, n_caption_lstm_step):
        self.dim_image = dim_image #4096
        self.n_words = n_words #2888
        self.lstm_size = lstm_size #256
        self.batch_size = batch_size #50 
        self.n_lstm_steps = n_lstm_steps  #???? 
        self.n_video_lstm_step=n_video_lstm_step #80
        self.n_caption_lstm_step=n_caption_lstm_step #20
       
        self.Wemb = tf.Variable(tf.random_uniform([n_words, lstm_size], -0.1, 0.1), name='Wemb')
        #self.bemb = tf.Variable(tf.zeros([dim_hidden]), name='bemb')
        
        self.lstm1 = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=False)
        self.lstm2 = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=False)
        """
        cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=False)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.4) 
        self.lstm1= tf.nn.rnn_cell.MultiRNNCell([cell] * 3)


        cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=False)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=0.4) 
        self.lstm2= tf.nn.rnn_cell.MultiRNNCell([cell] * 3)
        """
        
        self.encode_image_W = tf.Variable(tf.random_uniform([dim_image, lstm_size], -0.1, 0.1), name='encode_image_W')
        self.encode_image_b = tf.Variable(tf.zeros([lstm_size]), name='encode_image_b')

        self.embed_word_W = tf.Variable(tf.random_uniform([lstm_size, n_words], -0.1,0.1), name='embed_word_W')
        self.embed_word_b = tf.Variable(tf.zeros([n_words]), name='embed_word_b')
        
            
    def build_model(self):
        video = tf.placeholder(tf.float32, [self.batch_size, self.n_video_lstm_step, self.dim_image])
        caption = tf.placeholder(tf.int32, [self.batch_size, self.n_caption_lstm_step+1])
        caption_mask = tf.placeholder(tf.float32, [self.batch_size, self.n_caption_lstm_step+1])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b ) # (batch_size*n_lstm_steps, dim_hidden)
        image_emb = tf.reshape(image_emb, [self.batch_size, self.n_video_lstm_step, self.lstm_size])
        print(self.lstm1.state_size)
        state1 = tf.zeros([self.batch_size, self.lstm1.state_size])
        state2 = tf.zeros([self.batch_size, self.lstm2.state_size])
        # maybe attention could be added
        padding = tf.zeros([self.batch_size, self.lstm_size])# lstm_size =256

        probs = []
        loss = 0.0
       # attention_list = []

        ##############################  Encoding Stage ##################################
        for i in range(0, self.n_video_lstm_step):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            with tf.variable_scope("LSTM1",reuse=(i!=0)):
                output1, state1 = self.lstm1(image_emb[:,i,:], state1)

            with tf.variable_scope("LSTM2",reuse=(i!=0)):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)
                #attention_list.append(output2)
               
        ############################# Decoding Stage ######################################
        with tf.variable_scope(tf.get_variable_scope()):    
            for i in range(0, self.n_caption_lstm_step): ## Phase 2 => only generate captions
                if i == 0:
                    current_embed = tf.zeros([self.batch_size, self.lstm_size])
                else:
                    with tf.device("/cpu:0"):
                        current_embed = tf.nn.embedding_lookup(self.Wemb, caption[:,i-1])
    
                tf.get_variable_scope().reuse_variables()
                with tf.variable_scope("LSTM1"):
                    output1, state1 = self.lstm1(padding, state1)
    
                with tf.variable_scope("LSTM2"):
                    output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)
                
                ## attention mechanism here
                
                """ 
                attention_max_index = []
                #print(output2)
                #print(attention_list[1])
                for attention_vec in attention_list:
                    attention_max_index.append(tf.reduce_sum(output2*attention_vec, 1))
                output3 = tf.argmax(attention_max_index, axis = 0)
                ##
                for i in range(batch_size):
                   output2 = tf.add(output2[i,:], attention_list[output3[i]][i,:])
                
                ###
                """
                labels = tf.expand_dims(caption[:, i+1], 1) #to the next timestep input
                indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
                concated = tf.concat([indices, labels],1)
                onehot_labels = tf.sparse_to_dense(concated, tf.stack([self.batch_size, self.n_words]), 1.0, 0.0)
    
                logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b) 
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit_words, labels=onehot_labels)
                cross_entropy = cross_entropy * caption_mask[:,i]
                probs.append(logit_words)
    
                current_loss = tf.reduce_sum(cross_entropy)/self.batch_size
                tf.get_variable_scope().reuse_variables()

                #print(current_loss)
                loss = loss + current_loss
        print("finish Model build")
        return loss, video, caption, caption_mask, probs

    def build_generator(self):
        video = tf.placeholder(tf.float32, [1, self.n_video_lstm_step, self.dim_image])

        video_flat = tf.reshape(video, [-1, self.dim_image])
        image_emb = tf.nn.xw_plus_b(video_flat, self.encode_image_W, self.encode_image_b)
        image_emb = tf.reshape(image_emb, [1, self.n_video_lstm_step, self.lstm_size])

        state1 = tf.zeros([1, self.lstm1.state_size])
        state2 = tf.zeros([1, self.lstm2.state_size])
        padding = tf.zeros([1, self.lstm_size])

        generated_words = []

        probs = []
        embeds = []
        attention_list = []
        for i in range(self.n_video_lstm_step):
            if i > 0: tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(image_emb[:,i,:], state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([padding, output1], 1), state2)
                #attention_list.append(output2)
        for i in range(self.n_caption_lstm_step):

            tf.get_variable_scope().reuse_variables()

            if i == 0:
                current_embed = tf.zeros([1, self.lstm_size])

            with tf.variable_scope("LSTM1"):
                output1, state1 = self.lstm1(padding, state1)

            with tf.variable_scope("LSTM2"):
                output2, state2 = self.lstm2(tf.concat([current_embed, output1], 1), state2)
            
            ## attention mechanism here
            """
            attention_max_index = []
            print(output2)
            print(attention_list[1])
            for attention_vec in attention_list:
                attention_max_index.append(tf.reduce_sum(output2*attention_vec, 1))
            output3 = tf.argmax(attention_list, axis = 0)
            ##
            output2 += output3
            """
            logit_words = tf.nn.xw_plus_b(output2, self.embed_word_W, self.embed_word_b)
            
            
            max_prob_index = tf.argmax(logit_words, 1)[0]
            generated_words.append(max_prob_index)
            probs.append(logit_words)

            with tf.device("/cpu:0"):
                current_embed = tf.nn.embedding_lookup(self.Wemb, max_prob_index)
                current_embed = tf.expand_dims(current_embed, 0)

            embeds.append(current_embed)

        return video, generated_words, probs, embeds
        


def train():
    # we do the one hot encoding only so that softmax could be applied directly.     
    
    mapping_dict = mapping_training_data_dic()
    tf.reset_default_graph()
    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(wordtoindex),
            lstm_size=lstm_size,
            n_lstm_steps = n_video_lstm_step,
            batch_size=batch_size,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step) #no bias used here 
    
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        tf_loss, tf_video, tf_caption, tf_caption_mask, tf_probs = model.build_model()
    sess = tf.InteractiveSession()
        
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        train_op = tf.train.AdamOptimizer(lr_rate).minimize(tf_loss)     
        # train_op = tf.train.GradientDescentOptimizer(lr_rate).minimize(tf_loss)
        # train_op = tf.train.AdamOptimizer(lr_rate).minimize(tf_loss)
    
    tf.global_variables_initializer().run()
    saver = tf.train.Saver(max_to_keep=300, write_version=1)

    loss_fd = open('loss.txt', 'w')
    loss_record = []
    current_video_features = np.zeros([batch_size, 80, 4096])
    
        
    for epoch in range(n_epochs):
        loss_to_draw_epoch = []
            
        shuffle_list = [i for i in range(len(training_data_vec))]
        random.seed(epoch) # the seed is the epoch number
        random.shuffle(shuffle_list)
        for i in range(math.floor(total_training_samples/batch_size)):
            current_video_captions = []
            
            for j in range(batch_size):    
                current_video_captions.append(training_data_vec[shuffle_list[i*batch_size+j]])
                current_video_features[j,:] = train_X[mapping_dict[shuffle_list[i*batch_size+j]]]
            
            current_caption_matrix = sequence.pad_sequences(current_video_captions, padding='post', maxlen=n_caption_lstm_step)
            current_caption_matrix = np.hstack( [current_caption_matrix, np.zeros( [len(current_caption_matrix), 1] ) ] ).astype(int)
            current_caption_masks = np.zeros( (current_caption_matrix.shape[0], current_caption_matrix.shape[1]) )
            nonzeros = np.asarray(list(map(lambda x: (x != 0).sum() + 1, current_caption_matrix )))
    
            for ind, row in enumerate(current_caption_masks):
                row[:nonzeros[ind]] = 1
            #print(current_caption_masks)
            """
            probs_val = sess.run(tf_probs, feed_dict={
                tf_video: current_video_features,
                tf_caption: current_caption_matrix
                })
            """
            
            _, loss_val = sess.run(
                    [train_op, tf_loss],
                    feed_dict={
                        tf_video: current_video_features,
                        tf_caption: current_caption_matrix,
                        tf_caption_mask: current_caption_masks
                        })
        loss_to_draw_epoch.append(loss_val)
    
        print(loss_val)
    
        if np.mod(epoch, 7) == 0:
            print("Epoch ", epoch, " is done. Saving the model ...")
            saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)
            
                
                
            # write validation here
            
    return(loss_to_draw_epoch)

    # write testing here 
    # how to improve the model


def test(model_path='./model-609'):
    tf.reset_default_graph()
    test_videos = test_features
    model = Video_Caption_Generator(
            dim_image=dim_image,
            n_words=len(indextoword),
            n_lstm_steps=n_video_lstm_step,
            lstm_size=lstm_size,
            batch_size=batch_size,
            n_video_lstm_step=n_video_lstm_step,
            n_caption_lstm_step=n_caption_lstm_step)

    video_tf, caption_tf, probs_tf, last_embed_tf = model.build_generator()

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    test_output_txt_fd = open('S2VT_results.txt', 'w')
    
    video_feat = np.zeros([1,80,4096])
    output = []
    for i in range(test_videos.shape[0]):
        
        video_feat[0,:] = test_videos[i]
        #video_feat = np.load(video_feat_path)
        #video_mask = np.ones((video_feat.shape[0], video_feat.shape[1]))
        generated_word_index = sess.run(caption_tf, feed_dict={video_tf:video_feat})
        #print(generated_word_index)
        generated_words = [indextoword[i] for i in generated_word_index]
        punctuation = np.argmax(np.array(generated_words) == 'EOS') + 1
        if punctuation==1:
            punctuation = 6
        generated_words = generated_words[0:punctuation]
        

        generated_sentence = ' '.join(generated_words)
        generated_sentence = generated_sentence.replace('BOS', '')
        generated_sentence = generated_sentence.replace('PAD', '')
        generated_sentence = generated_sentence.replace('EOS', '')
        print(testing_order[i] + ',' + generated_sentence + '\n')
        test_output_txt_fd.write(testing_order[i] +','+ generated_sentence + '\n')



    test_output_txt_fd.close()    

    with open('S2VT_results.txt','r') as f:
        counter = 0
        for line in f:
            line = line.split()
            #line.insert( 1 ,'man')

            temp = [] 
            for i in range(len(line)):
                #if line[i] not in temp and line[i]!='a' :
                if line[i] not in temp :    
                    temp.append(line[i])

            done = ''
            if len(temp) > 8:
                for i in range(int(len(temp)/2)):
                    done += temp[i]
                    done += ' '
            else:
                for i in range(len(temp)):
                    done += temp[i]
                    done += ' '
            output.append(done)
    o = open(sys.argv[2] , 'w')
    for i in output:
        o.writelines(i+'\n')

#test()

train()
# build the LSTM model with either beam reduction or attention 
