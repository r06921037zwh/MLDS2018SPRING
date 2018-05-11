# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 17:54:04 2018

@author: zhewei
"""

import os
import pickle
import random
import jieba
jieba.set_dictionary('dict.txt.big.txt')

padToken, unknownToken, bosToken, eosToken = 0, 1, 2, 3

class Batch:
    #batch class for input usage
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []


def loadDataset(filename):
    """
    Load dataset
    : param filename: file's path
    : return word2id(dict), id2word(dict), trainSamples(list)
    
    which contains:
    1.word2id : word to id
    2.id2word : id to word
    3.trainSamples: list with Question and Answer
    """
    dataset_path = os.path.join(filename)
    print('Loading dataset from {}'.format(dataset_path))
    with open(dataset_path, 'rb') as handle:
        data = pickle.load(handle)
        word2id = data['word2id']
        id2word = data['id2word']
        trainSamples = data['trainSample']

    return word2id, id2word, trainSamples

def createBatch(samples):
    '''
    According samples add padding 
    : param samples: a batch's sample (list), each one in [question， answer] form
    : return: batch sample that can be directly feed in placeholder 
    '''
    batch = Batch()
    batch.encoder_inputs_length = [len(sample[0]) for sample in samples]
    batch.decoder_targets_length = [len(sample[1]) for sample in samples]

    max_source_length = max(batch.encoder_inputs_length) 
    max_target_length = max(batch.decoder_targets_length) 

    for sample in samples:
        #questions and PAD to max len of batch
        source = sample[0]
        pad = [padToken] * (max_source_length - len(source))
        batch.encoder_inputs.append(source + pad)

        #answers and PAD to max len of batch
        target = sample[1]
        pad = [padToken] * (max_target_length - len(target))
        batch.decoder_targets.append(target + pad)

    return batch

def getBatches(data, batch_size, size_of_data):
    '''
    divide data into different batch sample with batch_size。
    Then use createBatch on batch sample
    :param data: trainSamples(In loadDataset)，Q&A list
    :param batch_size: batch size
    :size_of_data: random size of data
    :return: a list with trainSamples in batch size, can be feed in feed_dict
    '''
    #shuffle before every epoch
    random.shuffle(data)
    data = data[:size_of_data]
    batches = []
    data_len = len(data)
    def genNextSamples():
        for i in range(0, data_len, batch_size):
            yield data[i:min(i + batch_size, data_len)]

    for samples in genNextSamples():
        batch = createBatch(samples)
        batches.append(batch)
    return batches

def sentence2enco(sentence, word2id):
    '''
    when testing, we should transfer input sentence into id, then use createBatch to process
    :param sentence: input sentence
    :param word2id: word2id dictionary
    :return: data that can directly feed in model
    '''
    if sentence == '':
        return None

    # tokenization
    #tokens = jieba.cut(sentence, cut_all=False)
    tokens = sentence
    words = [word for word in tokens]
    
    
    # word2id
    wordIds = []
    for word in words:
        #print("token {}".format(token))
        item = word2id.get(word, unknownToken)
        if item == 1:    
            item = random.randint(0,10000)
        wordIds.append(item)
    #filter_words = []
    '''
    for word in wordIds:
        if word != ('<UNK>'):
            filter_words.append(word)
    '''
    #wordIds = wordIds[:15]
    #createBatch
    batch = createBatch([[wordIds, []]])
    return wordIds, batch




response_pool = ["我是我誰呢?", "他們看來電"]