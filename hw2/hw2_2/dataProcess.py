# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:11:44 2018

@author: zhewei
"""
import os
import pickle
from collections import Counter

def make_dir(path):
    """ Create a directory if there isn't one already. """
    try:
        os.mkdir(path)
    except OSError:
        pass
    
class file_preprocessor(object):
    def __init__(self):
        self.data_path = 'data'
        self.corpus_name = 'clr_conversation.txt'
        self.preprocessed_path = 'processed_data_1'
        self.vocab_size = 50000
        
        self.questions = []
        self.answers = []
        
        self.train_enc = []
        self.train_dec = []
        
        self.train_enc_ids = []
        self.train_dec_ids = []
        
        self.vocab_list = []
        self.word2id = {}
        self.id2word = {}
        
        self.keywords = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
    
    def get_corpus(self):
        print("Reading corpus ... ")
        file_path = os.path.join(self.data_path, self.corpus_name)
        corpus = []
        with open(file_path, 'r', encoding='utf8') as fin:
            for line in fin:
                corpus.append(line.split('\n')[0])   
        return corpus

    def make_question_answer(self):
        '''pair the question and answer'''
        
        print("Making corpus Question-Answer form... ")
        self.corpus = self.get_corpus()
        # find the separate "+++$+++" position
        sep_idx = [idx for idx, _ in enumerate(self.corpus) if self.corpus[idx]=='+++$+++' ]
        sep_idx.insert(0, int(-1))
        
        # make the Q-A corpus
        for parts in range(len(sep_idx) - 1):
            part_min = sep_idx[parts]
            part_max = sep_idx[parts + 1]
            for line_id in range(part_min + 1, part_max): 
                question_idx = line_id
                answer_idx = line_id + 1
                if answer_idx >= part_max:
                    break
                else:
                    self.questions.append(self.corpus[question_idx])
                    self.answers.append(self.corpus[answer_idx])

        # split sentence into words (list)
        # append <EOS> to all answers
        for i in range(len(self.questions)):
            # only keep the sentence len < 10
            if(len(self.questions[i].split(' ')) <= 10) and (len(self.answers[i].split(' ')) <= 10):
                self.train_enc.append(self.questions[i].split(' '))
                self.train_dec.append(self.answers[i].split(' ') + ['<EOS>'])

    def write_dataset(self):
        '''write questions and answers to train.enc , train.dec'''
        print("Saving the dataset... ")
         # create path to store all the train & test encoder & decoder
        make_dir(self.preprocessed_path)
        
        # write to outfile
        filenames = ['train.enc', 'train.dec']
        files = []
        for filename in filenames:
            files.append(open(os.path.join(self.preprocessed_path, filename),'w', encoding='utf8'))
    
        for i in range(len(self.questions)):
            files[0].write(self.questions[i] + ' \n')
            files[1].write(self.answers[i] + ' \n')
    
        for file in files:
            file.close()
    
    def make_dict(self):
        print("Making dictionary ...")
        vocabs = []
        for line in self.questions:
            vocabs.extend(line.split(' '))
            
        counter = Counter(vocabs).most_common(self.vocab_size)
        vocab_list = [w for w, _ in counter]
        self.vocab_list = self.keywords + vocab_list
        
        print("Writing to vocab ...")
        vocab_path = os.path.join(self.preprocessed_path, 'vocab')       
        with open(vocab_path, 'w', encoding='utf8') as fout:
            for line in self.vocab_list:
                fout.write(line)
                fout.write('\n')
                
    def word_to_id(self):
        print("Translating word to id ...")
        self.word2id = {k: v for v, k in list(enumerate(self.vocab_list))}
        self.id2word = {k: v for k, v in list(enumerate(self.vocab_list))}
        
        filenames = ['train.enc', 'train.dec']
        contents = [self.train_enc, self.train_dec]
        for idx in range(len(filenames)):
            content_id = []
    
            print("Translating {} file ...".format(filenames[idx]))
            for line in contents[idx]:
                tmp=[]
                for word in line:
                    if word not in self.word2id:
                        tmp.append(self.word2id['<UNK>'])
                    else:
                        tmp.append(self.word2id[word])
                content_id.append(tmp)
            
            print("Writing to {} file ...".format(filenames[idx]))
            outfile_path =  os.path.join(self.preprocessed_path, 'id_' + filenames[idx])
            with open(outfile_path, 'w', encoding='utf8') as fout:
                for line in content_id:
                    for word in line:
                        fout.write(str(word) + ' ')
                    fout.write('\n')
            print("{} completed ".format(filenames[idx]))
            
            if idx == 0: self.train_enc_ids = content_id
            elif idx == 1: self.train_dec_ids = content_id
    
    # if a sentence comprises more a half <UNK> then drop it           
    def sift_unk_sent(self):
        enc_ids = []
        dec_ids = []
        
        for idx, line in enumerate(self.train_enc_ids):
            count = 0
            for ele in line:
                if ele == 1:
                    count += 1
            if count/len(line) < 0.5:
                enc_ids.append(self.train_enc_ids[idx])
                dec_ids.append(self.train_dec_ids[idx])
                
        self.train_enc_ids, self.train_dec_ids = enc_ids, dec_ids
                
    def save_file(self):
        print("Dumping trainFile ...")
        self.train_samples = [[q, a] for q, a in zip(self.train_enc_ids, self.train_dec_ids)]
        content = {'word2id': self.word2id, 
                   'id2word': self.id2word,
                   'trainSample': self.train_samples}
        
        dumpPath = os.path.join(self.preprocessed_path, 'trainFile.pkl')
        pickle.dump(content, open(dumpPath, 'wb'))
    
    
if __name__ == "__main__":
    fp = file_preprocessor()        
    fp.make_question_answer()
    fp.write_dataset()
    fp.make_dict()
    fp.word_to_id()
    fp.sift_unk_sent()
    fp.save_file()
    
    '''
    questions = fp.questions
    answers = fp.answers[:5]
    train_dec = fp.train_dec[:10]
    train_dec_id = fp.train_dec_ids[:10]
    a = fp.questions
    a = fp.train_enc
    a = fp.train_samples[:5]
    a = "我 很 喜歡 你 呢"
    a = a.split(' ')[:20]
    '''