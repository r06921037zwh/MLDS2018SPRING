# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 18:04:55 2018

@author: zhewei
"""

import tensorflow as tf
#from tensorflow.python.util import nest

class Seq2SeqModel():
    def __init__(self, rnn_size, num_layers, embedding_size, learning_rate, learning_rate_decay_factor,
                 word_to_idx, mode, use_attention, beam_search, beam_size, max_gradient_norm):
        
        self.learning_rate = tf.Variable(learning_rate, trainable=False, dtype=tf.float16)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.word_to_idx = word_to_idx
        self.vocab_size = len(self.word_to_idx)
        
        self.mode = mode    
        self.use_attention = use_attention            # use attention
        self.beam_search = beam_search                # using beam_search
        self.beam_size = beam_size
        self.max_gradient_norm = max_gradient_norm    # using gradient clip
                  
        self.build_model()                            # build model
    # create single rnn cell and wrap in a DropoutWrapper, feed in MultiRNNCell
    def _create_rnn_cell(self):
        def single_rnn_cell():
            #single_cell = tf.contrib.rnn.LSTMCell(self.rnn_size)
            single_cell = tf.contrib.rnn.GRUCell(self.rnn_size)
            cell = tf.contrib.rnn.DropoutWrapper(single_cell, output_keep_prob=self.keep_prob_placeholder)
            return cell
        # wrap in multiRNNCell
        cell = tf.contrib.rnn.MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return cell
    
    # build encoder-decoder model
    def build_model(self):
        print('building model ...')
        #=================================1, define placeholders
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')

        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
        self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
        
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.keep_prob_placeholder = tf.placeholder(tf.float32, name='keep_prob_placeholder')
        
        self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
        self.mask = tf.sequence_mask(self.decoder_targets_length, self.max_target_sequence_length, dtype=tf.float32, name='masks')

        #=================================2, define encoder
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            # define 2 layers LSTM with dropout
            encoder_cell = self._create_rnn_cell()
            
            # embedding , which is commonly used by encoder and decoder
            embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size], dtype=tf.float32)
            encoder_inputs_embedded = tf.nn.embedding_lookup(embedding, self.encoder_inputs)
            # use dynamic_rnn construct LSTM model
            # encoder_outputs use for attention，[batch_size*encoder_inputs_length*rnn_size],
            # encoder_state as decoder's initialization，[batch_size*rnn_szie]
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, 
                                                               encoder_inputs_embedded,
                                                               sequence_length=self.encoder_inputs_length, 
                                                               dtype=tf.float32)

        # =================================3, define decoder
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            encoder_inputs_length = self.encoder_inputs_length
            # if self.beam_search:
            #     # if use beam_search，we should do tile_batch on decoder's output，(copy by beam_size)。
            #     print("use beamsearch decoding..")
            #     encoder_outputs = tf.contrib.seq2seq.tile_batch(encoder_outputs, multiplier=self.beam_size)
            #     encoder_state = nest.map_structure(lambda s: tf.contrib.seq2seq.tile_batch(s, self.beam_size), encoder_state)
            #     encoder_inputs_length = tf.contrib.seq2seq.tile_batch(self.encoder_inputs_length, multiplier=self.beam_size)

            # we can use following two attentions
            #attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units=self.rnn_size, 
                                                                       #memory=encoder_outputs,
                                                                       #memory_sequence_length=encoder_inputs_length)
            attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units=self.rnn_size, memory=encoder_outputs, memory_sequence_length=encoder_inputs_length)
            
            # define LSTMCell for decoder，then Wrap it with attention wrapper
            decoder_cell = self._create_rnn_cell()
            decoder_cell = tf.contrib.seq2seq.AttentionWrapper(cell=decoder_cell, 
                                                               attention_mechanism=attention_mechanism,
                                                               attention_layer_size=self.rnn_size, 
                                                               name='Attention_Wrapper')
            
            #if use beam_seach, batch_size = self.batch_size * self.beam_size.
            #batch_size = self.batch_size if not self.beam_search else self.batch_size * self.beam_size
            #batch_size = self.batch_size
            
            #decoder's initial state is encoder's last hidden state
            decoder_initial_state = decoder_cell.zero_state(batch_size=self.batch_size, dtype=tf.float32).clone(cell_state=encoder_state)
            
            #output_layer = tf.layers.Dense(self.vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            
            if self.mode == 'train':
                # define decoder's input 
                # => append <BOS> at the beginning of decoder's target, delete <end> in the end, then embed it
                # decoder_inputs_embedded shape : [batch_size, decoder_targets_length, embedding_size]
                ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
                decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.word_to_idx['<BOS>']), ending], 1)
                decoder_inputs_embedded = tf.nn.embedding_lookup(embedding, decoder_input)
                
                #When training， use TrainingHelper(Next input is ground truth) + BasicDecoder(Wrapper) 
                training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_inputs_embedded,
                                                                    sequence_length=self.decoder_targets_length,
                                                                    time_major=False, 
                                                                    name='training_helper')
                
                training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, 
                                                                   helper=training_helper,
                                                                   initial_state=decoder_initial_state, 
                                                                   output_layer=tf.layers.Dense(self.vocab_size))
                #use dynamic_decode for decoding，decoder_outputs is a namedtuple:(rnn_outputs, sample_id)
                # rnn_output: [batch_size, decoder_targets_length, vocab_size]，vocab probability for every timestep，use it for loss
                # sample_id: [batch_size], tf.int32，final decoder answer
                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder)
                
                # calculate loss and gradient, define AdamOptimizer and train_op
                self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
                self.decoder_predict_train = tf.argmax(self.decoder_logits_train, axis=-1, name='decoder_pred_train')
                # use sequence_loss calculate loss (Mask: only those with digits=1, padding=0)
                self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.decoder_logits_train,
                                                             targets=self.decoder_targets, 
                                                             weights=self.mask)
    
                # Training summary for the current batch_loss
                tf.summary.scalar('loss', self.loss)
                self.summary_op = tf.summary.merge_all()
                
                # optimizer and clip_gradients
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                trainable_params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, trainable_params)
                clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))
                
            elif self.mode == 'decode':
                start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.word_to_idx['<BOS>']
                end_token = self.word_to_idx['<EOS>']
                # decoder phase: determine on whether using beam_search,
                # If beamsearch => BeamSearchDecoder（with in-builded helper class）
                # If not beamsearch => Greedy
                # Helper(Last max prob word as next input) + BasicDecoder
                if self.beam_search:
                    inference_decoder = tf.contrib.seq2seq.BeamSearchDecoder(cell=decoder_cell, embedding=embedding,
                                                                             start_tokens=start_tokens, end_token=end_token,
                                                                             initial_state=decoder_initial_state,
                                                                             beam_width=self.beam_size,
                                                                             output_layer=tf.layers.Dense(self.vocab_size))
                else:
                    decoding_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embedding,
                                                                               start_tokens=start_tokens, 
                                                                               end_token=end_token)
                    inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, 
                                                                        helper=decoding_helper,
                                                                        initial_state=decoder_initial_state,
                                                                        output_layer=tf.layers.Dense(self.vocab_size))
                    
                decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder, 
                                                                          maximum_iterations=15)
                # decode with dynamic_decode, decoder_outputs is a namedtuple
                
                # If not beam_search，decoder_outputs: (rnn_outputs, sample_id)
                # rnn_output: [batch_size, decoder_targets_length, vocab_size]
                # sample_id: [batch_size, decoder_targets_length], tf.int32

                # If beam_search，decoder_outputs: (predicted_ids, beam_search_decoder_output)
                # predicted_ids: [batch_size, decoder_targets_length, beam_size], with outcome
                # beam_search_decoder_output: BeamSearchDecoderOutput instance namedtuple(scores, predicted_ids, parent_ids)
                # return "predicted_ids" or "sample_id" for our last response
                if self.beam_search:
                    self.decoder_predict_decode = decoder_outputs.predicted_ids
                else:
                    self.decoder_predict_decode = tf.expand_dims(decoder_outputs.sample_id, -1)
                    
        # =================================4, save model
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, batch, showtime):
        #In training phase : self.train_op, self.loss, self.summary_op
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                      self.encoder_inputs_length: batch.encoder_inputs_length,
                      self.decoder_targets: batch.decoder_targets,
                      self.decoder_targets_length: batch.decoder_targets_length,
                      self.keep_prob_placeholder: 1.0,
                      self.batch_size: len(batch.encoder_inputs)}
        _, loss, summary = sess.run([self.train_op, self.loss, self.summary_op], feed_dict=feed_dict)
        
        if showtime % 100 == 0:
             Q, A, P = sess.run([self.encoder_inputs, self.decoder_targets, self.decoder_predict_train],
                           feed_dict=feed_dict)
             print(" ")
             print(Q[0])
             print(A[0])
             print(P[0])
        
        return loss, summary

    def eval(self, sess, batch):
        # In eval phase, no need backpropogation : self.loss, self.summary_op
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                      self.encoder_inputs_length: batch.encoder_inputs_length,
                      self.decoder_targets: batch.decoder_targets,
                      self.decoder_targets_length: batch.decoder_targets_length,
                      self.keep_prob_placeholder: 1.0,
                      self.batch_size: len(batch.encoder_inputs)}
        loss, summary = sess.run([self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    def infer(self, sess, batch):
        #In infer phase, we need only get response
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                      self.encoder_inputs_length: batch.encoder_inputs_length,
                      self.keep_prob_placeholder: 1.0,
                      self.batch_size: len(batch.encoder_inputs)}
        predict = sess.run([self.decoder_predict_decode], feed_dict=feed_dict)
        return predict
    
       