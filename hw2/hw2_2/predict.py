# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 21:32:32 2018

@author: zhewei
"""

import tensorflow as tf
from dataLoad import loadDataset, getBatches, sentence2enco
from model import Seq2SeqModel
import sys
import numpy as np
import os
import random


tf.app.flags.DEFINE_integer('rnn_size', 1024, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_size', 1024, 'Embedding dimensions of encoder and decoder inputs')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Learning rate')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Batch size')
tf.app.flags.DEFINE_integer('numEpochs', 3, 'Maximum # of training epochs')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 100, 'Save model checkpoint every this iteration')
#tf.app.flags.DEFINE_string('model_dir', 'model/', 'Path to save model checkpoints')
#tf.app.flags.DEFINE_string('model_name', 'model.ckpt', 'File name used for model checkpoints')
FLAGS = tf.app.flags.FLAGS

data_path = os.path.join('processed_data', 'trainFile.pkl')
word2id, id2word, trainSamples = loadDataset(data_path)

def predict_ids_to_seq(predict_ids, id2word, beam_szie):
    '''
    将beam_search返回的结果转化为字符串
    :param predict_ids: 列表，长度为batch_size，每个元素都是decode_len*beam_size的数组
    :param id2word: vocab字典
    :return:
    '''
    '''
    for single_predict in predict_ids:
        for i in range(beam_szie):
            predict_list = np.ndarray.tolist(single_predict[:, :, i])
            predict_seq = [id2word[idx] for idx in predict_list[0]]
            print(" ".join(predict_seq))
    '''
    #words = []
    response_pool = ["我不", "你是你的", "我真的", "有可能嗎", "太", "他", "他們好", "可以", "怎麼可以", "什麼", "的嗎", "好"]
    response = ""
    array_id = np.array(predict_ids).reshape(-1)
    #print(len(array_id))
    for predict_id in array_id:
        if predict_id not in (0, 1, 2, 3):
            word = id2word.get(predict_id)
            response += word
    #print(response)
    if response == "":
        response = response_pool[random.randint(0,9)]
    return response
    
def read_input(filename):
    sentences = []
    with open(filename, 'r', encoding='utf8') as fin:
        for line in fin:
            sentences.append(line)
    return sentences


with tf.Session() as sess:
    model = Seq2SeqModel(FLAGS.rnn_size, FLAGS.num_layers, FLAGS.embedding_size, FLAGS.learning_rate,0.99, word2id,
                         mode='decode', use_attention=True, beam_search=False, beam_size=5, max_gradient_norm=5.0)
    model_dir = os.path.join('model')
    model.saver.restore(sess, os.path.join(model_dir, 'chatbot.ckpt'))
    responses = []
    sentences = read_input(sys.argv[1])
    wordIds = []
    with open(sys.argv[2], 'w', encoding='utf8') as fout:
        for sentence in sentences:
            wordId, batch = sentence2enco(sentence, word2id)
            predict_ids = model.infer(sess, batch)
            response = predict_ids_to_seq(predict_ids, id2word, 5)
            responses.append(response)
            wordIds.append(wordId)
            fout.write(str(response) + '\n')
            
    '''
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
        batch = sentence2enco(sentence, word2id)
        # 获得预测的id
        predicted_ids = model.infer(sess, batch)
        # print(predicted_ids)
        # 将预测的id转换成汉字
        predict_ids_to_seq(predicted_ids, id2word, 5)
        
        print("> ", end="")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
    ''' 
        
        
        
