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


class generator():
    def __init__(self, filepath="/home/ylh/MCTS-RL/Ours/data", batch_size=16):
        self.base_dir = filepath
        self.text = os.listdir(filepath)  # 所有观测和标签
        np.random.shuffle(self.text)
        self.index = 0  # 注意位置
        self.count = len(self.text)
        self.batch_size = batch_size

    def next(self, batch_size=1):
        labels = []

        while True:
            self.index = self.index % self.count  # 1%300008=1
            if self.index == 0:  # 跑完一个轮回，重新打乱
                np.random.shuffle(self.text)
            text_path = self.text[self.index]
            self.index += 1
            # 读取.json格式文件的内容
            with open(text_path, 'r+') as file:
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
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 200  # num of simulations for each move
        self.c_puct = 5
        self.batch_size = 2048  # mini-batch size for training
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.random = random
        self.episode_len = 1500
        self.n_action_available = 25
        self.n_block = 1
        self.n_head = 1
        self.n_agent = 5
        self.n_embd = 128
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.n_block,self.n_embd,self.n_head,self.n_agent,
                                                   model_file=init_model,use_gpu=True)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.n_block,self.n_embd,self.n_head,self.n_agent,
                                                   use_gpu=True)

        self.mcts_player = MCTSPlayer(self.policy_value_net.get_actions,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_minibatch(self, file_path):
        batch_size = self.batch_size
        my_generator = generator(file_path, batch_size)
        return my_generator.next()

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = self.get_minibatch()
        state_batch = [data['obs'] for data in mini_batch]
        mcts_probs_batch = [data['prob'] for data in mini_batch]
        value_batch = [data['value'] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.get_actions(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    value_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.get_actions(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy))
        return loss, entropy

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                loss, entropy = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    self.policy_value_net.save_model('./current_policy_.model')
        except KeyboardInterrupt:
            print('\n\rquit')


if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
