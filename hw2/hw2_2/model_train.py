# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 10:59:27 2018

@author: zhewei
"""

import tensorflow as tf
from dataLoad import loadDataset,getBatches
from model import Seq2SeqModel
from tqdm import tqdm
import math
import os
import csv

class modelTrainer():
    def __init__(self, rnn_size, num_layers, embedding_size, learning_rate, learning_rate_decay_factor,
                 batch_size, numEpochs, steps_per_checkpoint, model_dir, model_name='chatbot.ckpt'):
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.batch_size = batch_size
        self.numEpochs = numEpochs
        self.steps_per_checkpoint = steps_per_checkpoint
        
        self.model_dir = model_dir
        self.model_name = model_name
        
        self.learning_rate = learning_rate
        self.learning_rate_decay_factor = learning_rate_decay_factor
                
        self.data_path = os.path.join('processed_data', 'trainFile.pkl')
        self.word2id, self.id2word, self.trainSamples = loadDataset(self.data_path)
    
    def write_loss(self, filename, index, loss):
        with open(filename, 'a', encoding='utf8') as handle:
            writer = csv.writer(handle, delimiter=',', lineterminator='\n')
            writer.writerow([index, loss])
            
    def train(self):   
        
        with tf.Session() as sess:
            model = Seq2SeqModel(rnn_size=self.rnn_size,
                                 num_layers=self.num_layers,
                                 embedding_size=self.embedding_size,
                                 learning_rate=self.learning_rate,
                                 learning_rate_decay_factor=self.learning_rate_decay_factor,
                                 word_to_idx=self.word2id, 
                                 mode='train',
                                 use_attention=True,
                                 beam_search=False,
                                 beam_size=5,
                                 max_gradient_norm=5.0)
        
            # if having pretrained model, use it
            # or, train a new one
            ckpt = tf.train.get_checkpoint_state(self.model_dir)  
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                pass     
                print('Reloading model parameters..')
                model.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                print('Created new model parameters..')
                sess.run(tf.global_variables_initializer())
        
            current_step = 0
            summary_writer = tf.summary.FileWriter(self.model_dir, graph=sess.graph)
            sess.run(model.learning_rate.assign(0.7))    
            showtime = 1
            
            index = 1
            for e in range(self.numEpochs):
                print("======= Epoch {}/{} =======".format(e + 1, self.numEpochs))
                batches = getBatches(self.trainSamples, self.batch_size, 10000)
                
                
                # Tqdm: process reminder, need to implement tqdm(iterator)。
                for nextBatch in tqdm(batches, desc="Training"):
                    loss, summary = model.train(sess, nextBatch, showtime)
                    current_step += 1
                    showtime += 1
                        
                    if current_step % 100 == 0:
                        tqdm.write("-- Step %d -- Loss %.2f " % (current_step, loss))
                                            
                    if current_step % 1000 == 0:
                        if (model.learning_rate.eval() > 0.001):
                            sess.run(model.learning_rate_decay_op)
                        tqdm.write("-- Learning Rate %.5f " % (model.learning_rate.eval()))
                        
                    # save model every steps_per_checkpoint
                    if current_step % self.steps_per_checkpoint == 0:
                        perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                        tqdm.write("-- Step %d -- Loss %.2f -- Perplexity %.2f -- LR %.5f" % (current_step, loss, perplexity, 
                                                                                              model.learning_rate.eval()))
                        summary_writer.add_summary(summary, current_step)

                        # save model
                        checkpoint_path = os.path.join('model', str(index)+'ePoch')
                        self.checkpoint_path = os.path.join(checkpoint_path, self.model_name)
                        print("Saving model named {}epoch  ...".format(index))
                        model.saver.save(sess, self.checkpoint_path)
                        print("model saved ... ")
                        
                        self.write_loss('loss.csv', index, loss)
                        index += 1 
    
         
            
if __name__ == "__main__":
     trainer = modelTrainer(rnn_size=1024, 
                           num_layers=2, 
                           embedding_size=1024, 
                           learning_rate=1.0, 
                           learning_rate_decay_factor=0.99,
                           batch_size=100, 
                           numEpochs=5000,
                           steps_per_checkpoint=2500,
                           model_dir=os.path.join('model'), 
                           model_name='chatbot.ckpt')     
     trainer.train()