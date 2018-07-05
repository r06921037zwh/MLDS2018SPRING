from agent_dir.agent import Agent
import os
import numpy as np
import tensorflow as tf
import csv


OBSERVATION_SIZE = 6400
UP_ACTION = 2
DOWN_ACTION = 3
action_dict = {DOWN_ACTION:0, UP_ACTION: 1}

def prepro(o):
    """
    (1)crop the original image
    (2)downsample the image and remove color
    (3)erase the background(two types)
    (4)everything else set 1 (paddle, ball)
    (5)80x80 -> 6400x1 (ravel())
    """    
    processed_obs = o[35:195]
    processed_obs = processed_obs[::2, ::2, 0]
    processed_obs[processed_obs == 109] = 0
    processed_obs[processed_obs == 144] = 0
    processed_obs[processed_obs != 0] = 1
    processed_obs = processed_obs.astype(np.float).ravel()
    return processed_obs

def discount_rewards(rewards, gamma):
    discounted_r = np.zeros_like(rewards)
    current_sum = 0
    for t in reversed(range(0, len(rewards))):
        if(rewards[t] != 0):
            current_sum = 0
        current_sum = current_sum * gamma + rewards[t]
        discounted_r[t] = current_sum
    return discounted_r

def write_episode_reward(filename, reward):
    with open(filename, 'a', encoding='utf-8') as fout:
        writer = csv.writer(fout, delimiter=',', lineterminator='\n')
        writer.writerow([float(reward)])

def identity(a):
    return a      

class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_PG,self).__init__(env)
        self.batch_size = 1
        self.checkpoint_per_episodes = 10
        self.discount_factor = 0.99
        self.smoothed_reward = None
        self.render = False
        self.env = env
        self.train_iter = 10000
        self.hidden_layer_size=50
        self.checkpoint_dir ='pong_pg'
        self.lr = tf.Variable(tf.constant(8e-6), dtype=tf.float32, name='learning_rate')
        self.lr_decay = self.lr.assign(self.lr * 0.95)
        self.sess = tf.InteractiveSession()
        ######################## Networks ########################
        # observation
        self.observation = tf.placeholder(tf.float32, [None, OBSERVATION_SIZE], 'observation')
        #self.observation = tf.placeholder(tf.float32, [None, 80, 80, 1], 'observation')
        
        # +1 for up, -1 for down
        self.sample_actions = tf.placeholder(tf.float32, [None, 1], 'sample_functions')
        self.advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')
        
        # hidden_layer
        h1 = tf.layers.dense(
                self.observation,
                units=200,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer())
        
        h2 = tf.layers.dense(
                h1,
                units=200,
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer())
        
        self.up_prob = tf.layers.dense(
                h2,
                units=1,
                activation=tf.sigmoid,
                kernel_initializer=tf.contrib.layers.xavier_initializer())
        
        ########################----------########################
        
        ######################## Loss and Optimizer ########################
        self.loss = tf.losses.log_loss(
                labels=self.sample_actions,
                predictions=self.up_prob,
                weights=self.advantage)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 'pg.ckpt')
        
        # Load saved model
        # self.load_checkpoint()
        
        # record episode reward in tensorboard
        self.episode_reward = tf.Variable(tf.constant(0.0), dtype=tf.float32, name='episode_reward')
        tf.summary.scalar('episode reward', self.episode_reward)
        
        # record loss in tensorboard
        tf.summary.scalar('loss', self.loss)
        self.merge_summary_op = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('log/pg_summary', self.sess.graph)
        ########################----------########################
        
        # If test : load_model
        if args.test_pg:
            print('Loading Trained Model ...')
            self.load_checkpoint()
    
    def save_checkpoint(self):
        print("Saving Checkpoint ...")
        self.saver.save(self.sess, self.checkpoint_file)
    
    def load_checkpoint(self):
        print("Loading Checkpoint ...")
        self.saver.restore(self.sess, self.checkpoint_file)
        
    def init_game_setting(self):
        self.batch_state_action_reward_tuples = []
        self.smooth_reward = None
        self.observation_memory = []
        
    def train_step(self, state_action_reward_tuples, global_step, reward_sum):
        print("Training with {} length s/a/r tuples".format(len(state_action_reward_tuples)))       
        states, actions, rewards = zip(*state_action_reward_tuples)
        states = np.vstack(states)
        actions = np.vstack(actions)
        rewards = np.vstack(rewards)
        self.sess.run(tf.assign(self.episode_reward, reward_sum))
        summary_str, loss, _= \
                self.sess.run([self.merge_summary_op, self.loss, self.train_op],
                              feed_dict={ 
                                      self.observation: states,
                                      self.sample_actions: actions,
                                      self.advantage: rewards})
    
        self.summary_writer.add_summary(summary_str, global_step)
        print("Loss {:3f}".format(loss))
        
    def train(self):
        global_step = 1
        train_batch = []
        for episode_n in range(self.train_iter):
            print("Episode {} ...".format(episode_n + 1))
            episode_done = None
            episode_reward_sum = 0
            round_n = 1      
            step_n = 1
            last_observation = prepro(self.env.reset())
            action = self.env.action_space.sample()
            observation, _, _, _ = self.env.step(action)
            observation = prepro(observation)
            n_win = 0
            n_lose = 0
            
            while not episode_done:
                if self.render:
                    self.env.render()
                observation_diff = observation - last_observation
                last_observation = observation
                up_prob = self.make_action_diff(observation_diff)
                #print("up prob {}".format(up_prob))
                #if np.random.uniform() < up_prob:
                if up_prob > 0.5:
                    action = UP_ACTION
                else:
                    action = DOWN_ACTION
                
                observation, reward, episode_done, info = self.env.step(action)
                observation = prepro(observation)
                episode_reward_sum += reward
                step_n += 1
                
                tup = (observation_diff, action_dict[action], reward)
                train_batch.append(tup)
                
                if reward == -1:
                    #print("Round {}: {} time steps; lost".format(round_n, step_n))
                    n_lose += 1
                if reward == +1:
                    #print("Round {}: {} time steps; win".format(round_n, step_n))
                    n_win += 1
                if reward == 0:
                    round_n += 1
                    step_n += 1
            print("Episode {} finished after {} rounds".format(episode_n + 1, round_n))
            print("Win: {}, Lose: {}, Episode reward: {}".format(n_win, n_lose, episode_reward_sum))
            write_episode_reward('episode_reward.csv', episode_reward_sum)
            
            #exponentially smooth the reward
            if self.smoothed_reward is None:
                self.smoothed_reward = episode_reward_sum
            else:
                self.smoothed_reward = self.smoothed_reward * 0.99 + episode_reward_sum * 0.01
            print("Reward Total was {:3f}, discounted moving avg of reward is {:3f}".format(episode_reward_sum, self.smoothed_reward))
            
            
            if episode_n % self.batch_size == 0:
                print("======= Training Episode {} =======".format(global_step))
                states, actions, rewards = zip(*train_batch)
                rewards = discount_rewards(rewards, self.discount_factor)
                rewards -= np.mean(rewards)
                rewards /= np.std(rewards)
                train_batch = list(zip(states, actions, rewards))
                self.train_step(train_batch, global_step, episode_reward_sum)
                train_batch = []
                global_step += 1
                
            if (episode_n + 1)% self.checkpoint_per_episodes == 0:
                self.save_checkpoint()
            
            if (episode_n + 1) % 500 == 0:
                self.sess.run(self.lr_decay)

    def make_action(self, observation, test=True):
        UP_ACTION = 2
        DOWN_ACTION = 3
        if self.observation_memory == []:
            init_observation = prepro(observation)
            action = self.env.get_random_action()
            second_observation, _, _, _ = self.env.step(action)
            second_observation = prepro(second_observation)
            observation_diff = second_observation - init_observation
            self.observation_memory = second_observation
            up_prob = self.make_action_diff(observation_diff)
            if np.random.uniform() < up_prob:
                action = UP_ACTION
            else:
                action = DOWN_ACTION
        else:
            observation = prepro(observation)
            observation_diff = observation - self.observation_memory
            self.observation_memory = observation
            up_prob = self.make_action_diff(observation_diff)
            #if np.random.uniform() < up_prob:
            if up_prob > 0.5:
                action = UP_ACTION
            else:
                action = DOWN_ACTION
        # action = self.env.get_random_action()
        return action
    
    def make_action_diff(self, observation, test=True):
        observation = np.array(observation).reshape(-1, 6400)
        feed_dict={self.observation: observation}
        up_prob = self.sess.run(self.up_prob, 
                                feed_dict=feed_dict)
        return up_prob