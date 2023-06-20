# -*- coding: utf-8 -*-
"""
Train the policy

@author: Liuhan Yin
"""

from __future__ import print_function
import random
import numpy as np
from collections import deque
import cv2
import os
import json
import sys
"""
from Module.mcts_withnet import MCTSPlayer
from Module.policy import PolicyNet as PolicyValueNet# Pytorch
"""
sys.path.insert(0,r'/home/ylh/MCTS-RL/CARLA')
sys.path.insert(0,r'/home/ylh/MCTS-RL/Ours')
from Module.mcts.mcts_withnet import MCTSPlayer
from Module.net.policy import TransformerPolicy as PolicyValueNet
import matplotlib.pyplot as plt


class generator():
    def __init__(self, filepath="/home/ylh/MCTS-RL/Ours/data", batch_size=16):
        self.base_dir = filepath
        with open(filepath + '/num.json', 'r+') as file:
            content = file.read()
        content = json.loads(content)  # 将json格式文件转化为python的字典文件
        self.num = content['num']  # 统计文件夹中的文件个数
        self.text = np.arange(self.num) # 所有观测和标签
        np.random.shuffle(self.text)
        self.index = 0  # 注意位置
        self.count = len(self.text)
        self.batch_size = batch_size

    def next(self):
        labels = []
        batch_size = self.batch_size
        while True:
            self.index = self.index % self.count  # 1%300008=1
            if self.index == 0:  # 跑完一个轮回，重新打乱
                np.random.shuffle(self.text)
            text_path = int(self.text[self.index])
            self.index += 1
            if (text_path % 5) == 0:
                continue
            # 读取.json格式文件的内容
            with open(self.base_dir + f'/data_file/{text_path}.json', 'r+') as file:
                content = file.read()

            label = json.loads(content)
            labels.append(label)

            if len(labels) == batch_size:
                self.index = 0
                break  # 一次batchsize完毕，推出循环

        return labels

    def test(self):
        labels = []
        batch_size = self.batch_size
        while True:
            self.index = self.index % self.count  # 1%300008=1
            if self.index == 0:  # 跑完一个轮回，重新打乱
                np.random.shuffle(self.text)
            text_path = int(self.text[self.index])
            self.index += 1
            if (text_path % 5) == 0:
                # 读取.json格式文件的内容
                with open(self.base_dir + f'/data_file/{text_path}.json', 'r+') as file:
                    content = file.read()

                label = json.loads(content)
                labels.append(label)

            if len(labels) == batch_size:
                self.index = 0
                break  # 一次batchsize完毕，推出循环

        return labels

    def __next__(self):
        return self.next(self.batch_size)

class TrainPipeline():
    def __init__(self, init_model=None):
        # training params
        self.learn_rate = 2e-5
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 200  # num of simulations for each move
        self.c_puct = 5
        self.batch_size = 2048  # mini-batch size for training
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.0005
        self.check_freq = 1000
        self.game_batch_num = 10000
        self.random = random
        self.n_action_available = 21
        self.n_block = 3
        self.n_head = 1
        self.n_agent = 2
        self.n_embd = 64
        self.obs_dim = 7
        self.action_dim = 21
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.n_block,self.n_embd,self.n_head,self.n_agent,
                                                   model_file=init_model,use_gpu=True)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.n_block,self.n_embd,self.n_head,self.n_agent,
                                                   use_gpu=True)

        #self.mcts_player = MCTSPlayer(self.policy_value_net.get_actions,
        #                              c_puct=self.c_puct,
        #                              n_playout=self.n_playout,
        #                              is_selfplay=1)

    def get_minibatch(self, file_path="/home/ylh/MCTS-RL/Ours/data"):
        batch_size = self.batch_size
        my_generator = generator(file_path, batch_size)
        return my_generator.next()

    def get_testbatch(self, file_path="/home/ylh/MCTS-RL/Ours/data"):
        batch_size = self.batch_size
        my_generator = generator(file_path, batch_size)
        return my_generator.test()

    def test(self):
        """update the policy-value net"""
        mini_batch = self.get_testbatch()
        state_batch = np.array([data['obs'] for data in mini_batch]).reshape((-1, self.n_agent, self.obs_dim))
        mcts_probs_batch = np.array([data['pro'] for data in mini_batch]).reshape((-1, self.n_agent, self.action_dim))
        value_batch = np.array([data['value'] for data in mini_batch]).reshape((-1, self.n_agent, 1))
        value_loss, policy_loss, entropy = self.policy_value_net.get_loss(
            state_batch,
            mcts_probs_batch,
            value_batch)

        return value_loss, policy_loss, entropy


    def policy_update(self):
        """update the policy-value net"""
        mini_batch = self.get_minibatch()
        state_batch = np.array([data['obs'] for data in mini_batch]).reshape((-1,self.n_agent,self.obs_dim))
        mcts_probs_batch = np.array([data['pro'] for data in mini_batch]).reshape((-1,self.n_agent,self.action_dim))
        value_batch = np.array([data['value'] for data in mini_batch]).reshape((-1,self.n_agent,1))
       # print(mini_batch)
        old_probs, old_v = self.policy_value_net.policy_value(state_batch,mcts_probs_batch)
      #  print(old_probs)
       # print(old_v)
        for i in range(self.epochs):
            value_loss,policy_loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    value_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch,mcts_probs_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
        explained_var_old = (1 -
                             np.var(np.array(value_batch) - old_v.flatten()) /
                             np.var(np.array(value_batch)))
        explained_var_new = (1 -
                             np.var(np.array(value_batch) - new_v.flatten()) /
                             np.var(np.array(value_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "value_loss:{},"
               "policy_loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        value_loss,
                        policy_loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return value_loss, policy_loss, entropy

    def run(self):
        """run the training pipeline"""
        vloss_ = []
        ploss_ = []
        vloss_test = []
        ploss_test = []
        try:
            for i in range(self.game_batch_num):
                value_loss, policy_loss, entropy = self.policy_update()
                vloss_.append(value_loss)
                ploss_.append(policy_loss)
                value_loss, policy_loss, entropy = self.test()
                vloss_test.append(value_loss)
                ploss_test.append(policy_loss)
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    self.policy_value_net.save(r'/home/ylh/MCTS-RL/Ours/model',i)

            plt.title('value loss curve', fontsize=15)  # 标题，并设定字号大小
            plt.xlabel(u'epoch', fontsize=10)  # 设置x轴，并设定字号大小
            plt.ylabel(u'loss', fontsize=10)  # 设置y轴，并设定字号大小
            vloss_ = np.array(vloss_)
            plt.plot(vloss_, color='#FF8C00',label="train")
            plt.plot(vloss_test, color='#4169E1', label="test")
            plt.legend(loc='best')  # 图例展示位置，数字代表第几象限3
            plt.show()

            plt.title('policy loss curve', fontsize=15)  # 标题，并设定字号大小
            plt.xlabel(u'epoch', fontsize=10)  # 设置x轴，并设定字号大小
            plt.ylabel(u'loss', fontsize=10)  # 设置y轴，并设定字号大小
            ploss_ = np.array(ploss_)
            plt.plot(ploss_, color='#FF8C00', label="train")
            plt.plot(ploss_test, color='#4169E1', label="test")
            plt.legend(loc='best')  # 图例展示位置，数字代表第几象限3
            plt.show()

        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    model = r'/home/ylh/MCTS-RL/Ours/model/3.pt'
    training_pipeline = TrainPipeline(init_model=None)
    training_pipeline.run()
