##############################   MLDS2018_Spring_HW4   ZhongYuan Li
##############################
##############################
from agent_dir.agent import Agent
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque
import random
import math
import numpy as np

######DDQN model
class DQN(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DQN,self).__init__()

        self.DNN_SIZE = 256
        self.ACTIONS = 3

        #4*84*84
        self.Cnn1 = nn.Conv2d(4 , 32 , kernel_size=8 , stride=4)
        self.Cnn1.weight.data.normal_(0, 0.1)
        #32*20*20 
        self.Cnn2 = nn.Conv2d(32 , 64 , kernel_size=4 , stride=2)
        self.Cnn2.weight.data.normal_(0, 0.1)
        #64*9*9 
        self.Cnn3 = nn.Conv2d(64 , 64 , kernel_size=3 , stride=1) 
        self.Cnn3.weight.data.normal_(0, 0.1)
        #64*7*7
        self.Flat = nn.Linear(64*7*7 , self.DNN_SIZE)
        self.Flat.weight.data.normal_(0, 0.1)
        self.Output = nn.Linear(self.DNN_SIZE , self.ACTIONS)
        self.Output.weight.data.normal_(0, 0.1)

    def forward(self, input):
        x = F.relu(self.Cnn1(input.transpose(3, 1).transpose(3, 2)))
        x = F.relu(self.Cnn2(x))
        x = F.relu(self.Cnn3(x))
        x = F.relu(self.Flat(x.view(x.size(0), -1)))
        return self.Output(x)

class Agent_DQN(Agent):
    def __init__(self, env, args):
        super(Agent_DQN,self).__init__(env)
        self.lr = 0.0001
        self.BATCH_SIZE = 32
        self.EPISODE = 50000
        self.EPSILON_MIN = 0.05
        self.EPSILON_MAX = 1.0
        self.EPSILON_STEPS = 1000000
        self.GAMMA = 0.99
        self.MEMORY_SIZE = 10000
        self.MEMORY = deque(maxlen=self.MEMORY_SIZE)
        self.Q = DQN().cuda()
        self.Q_target = DQN().cuda()
        self.REWARDS = []
        self.LOSS = []
        self.Q_target.load_state_dict(self.Q.state_dict())
        self.optimizer = torch.optim.RMSprop(self.Q.parameters(), lr=self.lr)

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')
            #self.Q.load_state_dict(torch.load('./MYmodels/dqn_43100.pt'))
            #self.Q.load_state_dict(torch.load('./DDQN_models/DDQN_33500.pt'))
            #self.Q.load_state_dict(torch.load('./DDQN_models/DDQN_{0}.pt'.format(str(model_num))))
            self.Q.load_state_dict(torch.load('./DDQN_33500.pt'))
            
        ##################
        # YOUR CODE HERE #
        ##################

        
    def Update_EPSILON(self , times):
        if times < self.EPSILON_STEPS:
            eps = self.EPSILON_MIN + (self.EPSILON_MAX - self.EPSILON_MIN) * ((self.EPSILON_STEPS - times) / self.EPSILON_STEPS)
        else:
            eps = 0
        return eps

    def Update(self):
        if len(self.MEMORY) < self.BATCH_SIZE:
            return 0
        batch = random.sample(self.MEMORY, self.BATCH_SIZE)
        _state, _next_state, _action, _reward, _done = zip(*batch)
        
        _state = Variable(torch.stack(_state)).cuda().squeeze()
        _next_state = Variable(torch.stack(_next_state)).cuda().squeeze()
        _action = Variable(torch.stack(_action)).cuda()
        _reward = Variable(torch.stack(_reward)).cuda()
        _done = Variable(torch.stack(_done)).cuda()
        
        q = self.Q(_state).gather(1, _action)

        ####  DDQN
        action_new = self.Q(_next_state).detach().max(-1)[1].unsqueeze(-1)
        Qs = self.Q_target(_next_state).detach()
        Q_value = Variable(torch.rand(32,1)).cuda()

        for i in range(self.BATCH_SIZE):
            Q_value[i] = Qs[i][action_new[i]]
        
        
        q_ = _reward + (1 - _done) * self.GAMMA * Q_value
        self.optimizer.zero_grad()

        loss = F.mse_loss(q, q_)
        loss.backward()
        self.optimizer.step()
        return loss.data[0]

    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        ##################
        # YOUR CODE HERE #
        ##################
        pass


    def train(self):
        """
        Implement your training algorithm here
        """
        ##################
        # YOUR CODE HERE #
        ##################
        step = 0
        memory_counter = 0
        for episode in range(1, self.EPISODE):
            state = self.env.reset()
            Total_reward = 0
            is_done = False
            loss = []
            for i in range(random.randint(1, 30)):
                state , _ , _ , _ = self.env.step(1)
            state = state.astype(np.float64)

            while not is_done:
                eps = self.Update_EPSILON(step)
                if random.random() < eps:
                    act = random.randint(0, 2)
                else:
                    act = self.make_action(state , False)

                next_state, reward, is_done, _ = self.env.step(act + 1)
                next_state = next_state.astype(np.float64)
                Total_reward += reward
                step += 1
                self.MEMORY.append((torch.FloatTensor([state]),torch.FloatTensor([next_state]),torch.LongTensor([act]), torch.FloatTensor([reward]),torch.FloatTensor([is_done])))
                
                if step >= 10000 and step % 5 == 0:
                    loss.append(self.Update())
                if step % 1000 == 0:
                    self.Q_target.load_state_dict(self.Q.state_dict())
                state = next_state

            self.LOSS.append(np.mean(loss))
            self.REWARDS.append(Total_reward)
            print('Episode: {0} , epsilon: {1} , REWARD: {2} , LOSS: {3}'.format(episode , eps , Total_reward , np.mean(loss)))
            
            if e % 50 == 0:
                print('Save data')
                torch.save(self.Q.state_dict(), './DDQN_models/DDQN_{0}.pt'.format(episode))
                np.save('./DDQN_models/DDQN_REWARDS' , np.array(self.REWARDS))
                np.save('./DDQN_models/DDQN_LOSS' , np.array(self.LOSS))

            


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)

        Return:
            action: int
                the predicted action from trained model
        """
        ##################
        # YOUR CODE HERE #
        ##################
        act = self.Q(Variable(torch.FloatTensor(observation).unsqueeze(0)).cuda()).max(-1)[1].data[0]
        if test:
            return act+1
        else:
            return act
