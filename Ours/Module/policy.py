# -*- coding: utf-8 -*-
"""
the policy net

@author: Liuhan Yin
"""


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable

def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Net(nn.Module):

    def __init__(self, vector_length, n_action_available):
        super(Net, self).__init__()

        #the dim of imput img and state vector
        self.vector_length = vector_length
        self.n_action_available = n_action_available

        """
        net structure
        img*--C--C--C--||--|--|--@policy
                       ||
            state#--|--||--|--|--$value
            state:ve,gama_e,vo,d,theta,gama_o 6d array
        """
        # common layers
        self.out_dim = 256
        self.conv1 = nn.Conv2d(3, 64, 4, stride=2, padding=1)#输入256*256
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1) #输出8*8

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)

        self.state_fc1 = nn.Linear(vector_length, 128)

        # action policy layers
        self.act_fc1 = nn.Linear(128 + self.out_dim*8*8, 1024)
        self.act_fc2 = nn.Linear(1024, n_action_available)

        # state value layers
        self.val_fc1 = nn.Linear(128 + self.out_dim*self.img_width*self.img_height, 64)
        self.val_fc2 = nn.Linear(64, 1)

    def forward(self, img_input, state_input):
        # common layers
        x = self.conv1(img_input)
        x = self.bn1(x)
        x = F.leaky_relu(x, inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, inplace=True)
        x = F.max_pool2d(x, 2, 2)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.leaky_relu(x, inplace=True)

        s = F.leaky_relu(self.state_fc1(state_input), inplace=True)

        """reshape and combine"""  
        x = x.view(-1, self.out_dim*8*8)
        x = torch.cat([x, s], 0)

        # action policy layers
        x_act = F.leaky_relu(self.act_fc1(x), inplace=True)
        x_logit = self.act_fc2(x_act)
        x_act = F.log_softmax(x_logit)

        # state value layers
        x_val = F.leaky_relu(self.val_fc1(x), inplace=True)
        x_val = F.tanh(self.val_fc2(x_val))
        return x_logit, x_act, x_val

class PolicyValueNet():
    """policy-value network """
    def __init__(self, vector_length, n_action_available,
                 model_file=None, use_gpu=False):
        self.use_gpu = use_gpu
        self.vector_length = vector_length
        self.n_action_available = n_action_available
        self.l2_const = 1e-4  # coef of l2 penalty
        # the policy value net module
        if self.use_gpu:
            self.policy_value_net = Net(vector_length, n_action_available).cuda()
        else:
            self.policy_value_net = Net(vector_length, n_action_available)
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)

        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, img_batch, state_batch):
        """
        input: a batch of states and imgs
        output: a batch of action probabilities and state values
        """
        if self.use_gpu:
            img_batch = Variable(torch.FloatTensor(np.array(img_batch)).cuda())
            state_batch = Variable(torch.FloatTensor(np.array(state_batch)).cuda())
            log_act_probs, value = self.policy_value_net(img_batch, state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return act_probs, value.data.cpu().numpy()
        else:
            img_batch = Variable(torch.FloatTensor(np.array(img_batch)))
            state_batch = Variable(torch.FloatTensor(np.array(state_batch)))
            log_act_probs, value = self.policy_value_net(img_batch, state_batch)
            act_probs = np.exp(log_act_probs.data.numpy())

            return act_probs, value.data.numpy()

    def policy_value_fn(self, img, state):
        """
        input: board
        output: a list of (action, probability) tuples for each available
        action and the score of the board state
        """
        legal_positions = np.arange(100)
        if self.use_gpu:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(img)).cuda().float(), Variable(torch.from_numpy(state)).cuda().float())
            act_probs = np.exp(log_act_probs.data.cpu().numpy().flatten())
        else:
            log_act_probs, value = self.policy_value_net(
                    Variable(torch.from_numpy(img)).float(), Variable(torch.from_numpy(state)).float())
            act_probs = np.exp(log_act_probs.data.numpy().flatten())
        act_probs = zip(legal_positions, act_probs[legal_positions])
        value = value.data[0][0]
        return act_probs, value

    def train_step(self, img_batch, state_batch, mcts_probs, value_batch, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            img_batch = Variable(torch.FloatTensor(np.array(img_batch)).cuda())
            state_batch = Variable(torch.FloatTensor(np.array(state_batch)).cuda())
            mcts_probs = Variable(torch.FloatTensor(np.array(mcts_probs)).cuda())
            value_batch = Variable(torch.FloatTensor(np.array(value_batch)).cuda())
        else:
            img_batch = Variable(torch.FloatTensor(np.array(img_batch)))
            mcts_probs = Variable(torch.FloatTensor(np.array(mcts_probs)))
            value_batch = Variable(torch.FloatTensor(np.array(value_batch)))

        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_net(img_batch, state_batch)
        # define the loss = (z - v)^2 - pi^T * log(p) + c||theta||^2
        # Note: the L2 penalty is incorporated in optimizer
        value_loss = F.mse_loss(value.view(-1), value_batch)
        policy_loss = -torch.mean(torch.sum(mcts_probs*log_act_probs, 1))
        loss = value_loss + policy_loss
        # backward and optimize
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        entropy = -torch.mean(
                torch.sum(torch.exp(log_act_probs) * log_act_probs, 1)
                )
        return loss.item(), entropy.item()

    def get_policy_param(self):
        net_params = self.policy_value_net.state_dict()
        return net_params

    def save_model(self, model_file):
        """ save model params to file """
        net_params = self.get_policy_param()  # get model params
        torch.save(net_params, model_file)
