# -*- coding: utf-8 -*-
"""
Train the policy

@author: Liuhan Yin
"""

from __future__ import print_function
import random
import numpy as np
from collections import defaultdict, deque
import sys
import cv2
import glob
import os
import json
"""
sys.path.append(r'/home/ylh/MCTS-Carla/scripts/Ours')
from Module.mcts_withnet import MCTSPlayer
from Module.policy import PolicyNet as PolicyValueNet# Pytorch
"""
from ..Module.policy import PolicyValueNet
from ..Module.mcts_withnet import MCTSPlayer


class generator():
    def __init__(self, filepath="/home/ylh/MCTS-RL/Ours/data", batch_size=16):  # 形参用到的，路径+batchsize+是否数据增强
        self.base_dir = filepath
        self.image_names = os.listdir(filepath + '/img')  # 所有图片名称
        np.random.shuffle(self.image_names)
        self.index = 0  # 注意位置
        self.count = len(self.image_names)
        self.batch_size = batch_size

    def next(self, batch_size=16):
        images = []
        labels = []

        while True:
            self.index = self.index % self.count  # 1%300008=1
            if self.index == 0:  # 跑完一个轮回，重新打乱
                np.random.shuffle(self.image_names)
            image_path = self.image_names[self.index]
            self.index += 1
            image = cv2.imread(image_path)
            if image is None:
                print("\n[WARRING]: 读取图片 '{}' 失败".format(image_path))
                continue
            images.append(image)  # list中存放各个图片的np格式

            state_path = image_path.split('img/')[0] + 'state/' + image_path.split('img/')[1].split('.')[0] + '.json'
            # 读取.json格式文件的内容
            with open(state_path, 'r+') as file:
                content = file.read()

            label = json.loads(content)
            labels.append(label)

            if len(images) == batch_size:
                break  # 一次batchsize完毕，推出循环

        images = np.asarray(images, dtype=np.uint8)  # 将list格式的images转为n h w c用于data_aug_sequential输入

        return images / 255, labels

    def __next__(self):
        return self.next(self.batch_size)

class TrainPipeline():
    def __init__(self, init_model=None):
        # training params
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 400  # num of simulations for each move
        self.c_puct = 5
        self.buffer_size = 10000
        self.batch_size = 2048  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 50
        self.game_batch_num = 1500
        self.best_win_ratio = 0.0
        self.random = random
        self.episode_len = 1500
        self.vector_length = 10
        self.n_action_available = 100
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = 1000
        if init_model:
            # start training from an initial policy-value net
            self.policy_value_net = PolicyValueNet(self.vector_length, self.n_action_available,
                                                   model_file=init_model,use_gpu=True)
        else:
            # start training from a new policy-value net
            self.policy_value_net = PolicyValueNet(self.vector_length, self.n_action_available,
                                                   use_gpu=True)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
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
        img_batch = [data[0] for data in mini_batch]
        d_batch = [data[1] for data in mini_batch]
        state_batch = [data['state'] for data in d_batch]
        mcts_probs_batch = [data['prob'] for data in d_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    img_batch,
                    state_batch,
                    mcts_probs_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
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
