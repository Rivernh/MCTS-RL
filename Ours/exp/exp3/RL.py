# -*- coding: utf-8 -*-
"""
MCTS for autonomous driving

@author: Liuhan Yin
"""

from __future__ import print_function
import sys

sys.path.insert(0, r'/home/ylh/MCTS-RL/CARLA')
sys.path.insert(0, r'/home/ylh/MCTS-RL/Ours')
from Module.mcts.mcts_withnet import MCTSPlayer as MCTS
import numpy as np
from gym_carla.envs import carla_env
from utils.process import start_process, kill_process
from kinematics.model import Env_model
from Module.net.policy import TransformerPolicy as PolicyValueNet
import matplotlib.pyplot as plt
import time as t
import json
import os
import json
from Module.net.policy import TransformerPolicy as PolicyValueNet


def Env_init():
    # parameters for the gym_carla environment
    params = {
        'number_of_vehicles': 0,
        'display_size': 256,  # screen size of bird-eye render
        'max_past_step': 2,  # the number of past steps to draw
        'dt': 0.05,  # time interval between two frames
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'opponent_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'port': 2000,  # connection port
        'town': 'Town03',  # which town to simulate
        'max_time_episode': 10000,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'obs_range': 32,  # observation range (meter)
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 10,  # desired speed (m/s)
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
        'display_route': False,  # whether to render the desired route
        'ego_transform': [ (165.5, -5.3, 180.2), (149, -28.2, 90),],
        'target_transform': [(96.3, -6.9, 180.2),(96.3, -6.9, 180.2)],
        'noise': False,
        'long':False,
        'temp_transform':[(-0.3,-155.6,94.2),(82.7,-145.7,92.1),(150.3,-146.1,93.2),(200.7,130.7,91.1),],
        'temp_target': [(-44,-4.2,176),(78.4,-60.7,92.1),(148,-50,93.2),(100.7,130.7,91.1),],
        'temp_destroy':[-35,58,118,-35]
    }
    env = carla_env.CarlaEnv(params)
    env.reset()
    return env

def run(thre,n,model):
    start_process(show=False)
    car_env = Env_init()  # carla env
    wpt = []
    ap_vehicle = []
    vehicle = car_env.egos  # 控制的智能体
    # vehicle.append(car_env.barrier) #控制的智能体
    time = []
    v = []
    target_v = []
    num = n
    count = num % 1722
    cnt = 0
    v1 = 0
    v2 = 0
    judge_1 = 0
    judge_2 = 0
    try:
        best_policy = PolicyValueNet(3,64,1,2,model_file=model,use_gpu=True)
        car = MCTS(best_policy.policy_value_fn,c_puct=0.2, n_playout=100)
        done = False
        car1_move = (0, 0)
        car2_move = (0, 0)
        move = [car1_move, car2_move]
        while not done:
            next_obs, _, done, _ = car_env.step(move)
            wpt.clear()
            wpt_temp = car_env.waypoints_all
            if len(wpt_temp[0]) < 8 or len(wpt_temp[1]) < 8:
                judge_1 = 1
                judge_2 = 1
                return True
            for i in range(len(wpt_temp)):
                wpt.append(wpt_temp[i][2:8])
            if car_env.time_step % 2 == 0:
                state = Env_model(vehicle, wpt, ap_vehicle, dt=0.1)  # 动力学仿真环境

                car1_move,p1 = car.get_action(state, 0)
                car2_move,p2 = car.get_action(state, 1)
                move = [car1_move, car2_move]
                #state.reset()
                #v1 += state.run_episode(0)
                #state.reset()
                #v2 += state.run_episode(1)
                #print(v1,v2)
                print(f"move:{move}")
                dic = {"obs":[state.UGV_info[0][0],state.UGV_info[0][1],state.UGV_info[0][2],state.speed_temp[0],
                              wpt[0][2][0],wpt[0][2][1],0] + [state.UGV_info[1][0], state.UGV_info[1][1], state.UGV_info[1][2], state.speed_temp[1],
                              wpt[1][2][0], wpt[1][2][1],1],
                       "pro":list(p1) + list(p2),
                       "value":[v1,v2]
                       }
                dict_json = json.dumps(dic)  # 转化为json格式文件
                # 将json文件保存为.json格式文件
                with open('/home/ylh/MCTS-RL/Ours/data_circle/data_file/' + f'{count}.json', 'w+') as file:
                    file.write(dict_json)
                #t.sleep(0.1)
                count += 1
                count = count % 1722
                cnt += 1
                del state
            # 绘图数据
            time.append(car_env.time_step * 0.05)
            v.append(car_env.v_now)
            target_v.append(move)

            if cnt > thre:
                done = True
        print("done!")
        judge_1 = -1
        judge_2 = -1


    except KeyboardInterrupt:
        print('\n\rquit')
    finally:
        car_env._set_synchronous_mode(False)
        kill_process()
        #judge_1 = input("1车是否通过：-1/0/1:")
        #judge_1 = int(judge_1)
        #judge_2 = input("2车是否通过：-1/0/1:")
        #judge_2 = int(judge_2)
        for i in range(cnt):
            path = '/home/ylh/MCTS-RL/Ours/data_circle/data_file/' + f'{(num+i)%1722}.json'
            with open(path, 'r+') as file:
                content = file.read()
                content = json.loads(content)  # 将json格式文件转化为python的字典文件
                pro = content['pro']
                value = content['value']
            obs = content['obs']
            dic = {"obs": obs,
                   "pro": pro,
                   "value": [judge_1,judge_2]
                   }
            dict_json = json.dumps(dic)  # 转化为json格式文件
            with open(path, 'w+') as file:
                file.write(dict_json)
        num = count
        dic = {"num": num
               }
        dict_json = json.dumps(dic)  # 转化为json格式文件
        # 将json文件保存为.json格式文件
        with open('/home/ylh/MCTS-RL/Ours/data_circle/num.json', 'w+') as file:
            file.write(dict_json)

        """
        plt.title('speed curve', fontsize=15)  # 标题，并设定字号大小
        plt.xlabel(u'time(s)', fontsize=10)  # 设置x轴，并设定字号大小
        plt.ylabel(u'speed(m/s)', fontsize=10)  # 设置y轴，并设定字号大小
        v = np.array(v)

        plt.plot(time, v[:, 0], color='#1E90FF', linewidth=1.0, linestyle='-', marker='o', label='ego agent1')
        plt.plot(time, v[:, 1], color='#6A5ACD', linewidth=1.0, linestyle='-', marker='o', label='ego agent2')
        # plt.plot(time, np.array(target_v)[:,0], color='#6A5ACD', linewidth=1.0, linestyle=':', marker='s', label='target speed')
        # plt.plot(time, np.array(target_v)[:,1], color='#708090', linewidth=1.0, linestyle=':', marker='s', label='target steer')
        plt.legend(loc='best')  # 图例展示位置，数字代表第几象限3
        plt.show()
        """


class generator():
    def __init__(self, filepath="/home/ylh/MCTS-RL/Ours/data_circle", batch_size=16):
        self.base_dir = filepath
        with open(filepath + '/num.json', 'r+') as file:
            content = file.read()
        content = json.loads(content)  # 将json格式文件转化为python的字典文件
        self.num = 1722#content['num']  # 统计文件夹中的文件个数
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
        self.learn_rate = 5e-4
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = 1.0  # the temperature param
        self.n_playout = 200  # num of simulations for each move
        self.c_puct = 5
        self.batch_size = 512  # mini-batch size for training
        self.play_batch_size = 1
        self.epochs = 5  # num of train_steps for each update
        self.kl_targ = 0.02
        self.check_freq = 100
        self.game_batch_num = 10000
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

    def get_minibatch(self, file_path="/home/ylh/MCTS-RL/Ours/data_circle"):
        batch_size = self.batch_size
        my_generator = generator(file_path, batch_size)
        return my_generator.next()

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        model_file = r'/home/ylh/MCTS-RL/Ours/model/current_policy.pt'
        for i in range(n_games):
            path = r'/home/ylh/MCTS-RL/Ours/data_circle/num.json'  # 输入文件夹地址
            with open(path, 'r+') as file:
                content = file.read()
            content = json.loads(content)  # 将json格式文件转化为python的字典文件
            num = content['num']  # 统计文件夹中的文件个数
            run(150,num,model_file)

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
           # if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
           #     break
        # adaptively adjust the learning rate
        #if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
        #    self.lr_multiplier /= 1.5
        #elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
         #   self.lr_multiplier *= 1.5
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
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                value_loss, policy_loss, entropy = self.policy_update()
                vloss_.append(value_loss)
                ploss_.append(policy_loss)
                # check the performance of the current model,
                # and save the model params
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    self.policy_value_net.save(r'/home/ylh/MCTS-RL/Ours/model',i)

            plt.title('loss curve', fontsize=15)  # 标题，并设定字号大小
            plt.xlabel(u'epoch', fontsize=10)  # 设置x轴，并设定字号大小
            plt.ylabel(u'loss', fontsize=10)  # 设置y轴，并设定字号大小
            vloss_ = np.array(vloss_)
            plt.plot(vloss_, label="value loss")
            plt.legend(loc='best')  # 图例展示位置，数字代表第几象限3
            plt.show()

            plt.title('loss curve', fontsize=15)  # 标题，并设定字号大小
            plt.xlabel(u'epoch', fontsize=10)  # 设置x轴，并设定字号大小
            plt.ylabel(u'loss', fontsize=10)  # 设置y轴，并设定字号大小
            ploss_ = np.array(ploss_)
            plt.plot(ploss_, label="policy loss")
            plt.legend(loc='best')  # 图例展示位置，数字代表第几象限3
            plt.show()

        except KeyboardInterrupt:
            print('\n\rquit')

if __name__ == '__main__':
    model_file=r'/home/ylh/MCTS-RL/Ours/model/current_policy.pt'
    training_pipeline = TrainPipeline(init_model=model_file)
    training_pipeline.run()
