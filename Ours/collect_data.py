# -*- coding: utf-8 -*-
"""
MCTS for autonomous driving

@author: Liuhan Yin
"""

from __future__ import print_function
import sys

sys.path.insert(0, r'/home/ylh/MCTS-RL/CARLA')
sys.path.insert(0, r'/home/ylh/MCTS-RL/Ours')
from Module.mcts.mcts_withnet import MCTSPlayer
import numpy as np
from gym_carla.envs import carla_env
from utils.process import start_process, kill_process
from kinematics.model import Env_model
import matplotlib.pyplot as plt
import time as t
import os
import json


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
        'ego_transform': [(165.5, -5.3, 180.2), (149, -28.2, 90), ],
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

def policy_value_fn(TreeEnv):
    """a function that takes in a obs and outputs a list of (action, probability)
    tuples and a score for the obs"""
    # return uniform probabilities and 0 score for pure MCTS
    availables = np.arange(21)
    action_probs = np.ones(len(availables))/len(availables)
    return zip(availables, action_probs), 0

def run(thre,n):
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
    count = num
    v1 = 0
    v2 = 0
    judge_1 = 0
    judge_2 = 0
    try:
        car = MCTSPlayer(policy_value_fn,c_puct=0.2, n_playout=500,is_selfplay=1)
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
                with open('./data/data_file/' + f'{count}.json', 'w+') as file:
                    file.write(dict_json)
                #t.sleep(0.1)
                count += 1
                """
                dic = {"x": state.UGV_info[1][0],
                       "y": state.UGV_info[1][1],
                       "yaw": state.UGV_info[1][2],
                       "speed": state.speed_temp[1],
                       "goal_x": wpt[1][2][0],
                       "goal_y": wpt[1][2][1],
                       "obs": [state.UGV_info[1][0], state.UGV_info[1][1], state.UGV_info[1][2], state.speed_temp[1],
                               wpt[1][2][0], wpt[1][2][1]],
                       "pro":list(p2),
                       "value":v2
                       }
                dict_json = json.dumps(dic)  # 转化为json格式文件
                # 将json文件保存为.json格式文件
                with open('./data/data_file/' + f'{count}.json', 'w+') as file:
                    file.write(dict_json)
                #t.sleep(0.1)
                count += 1
                """
                del state
            # 绘图数据
            time.append(car_env.time_step * 0.05)
            v.append(car_env.v_now)
            target_v.append(move)

            if count - num > thre:
                done = True
        print("done!")
        judge_1 = -1
        judge_2 = -1


    except KeyboardInterrupt:
        print('\n\rquit')
    finally:
        judge_1 = input("1车是否通过：-1/0/1:")
        judge_1 = int(judge_1)
        judge_2 = input("2车是否通过：-1/0/1:")
        judge_2 = int(judge_2)
        for i in range(count-num):
            path = './data/data_file/' + f'{num+i}.json'
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
        car_env._set_synchronous_mode(False)
        kill_process()
        num = count
        dic = {"num": num
               }
        dict_json = json.dumps(dic)  # 转化为json格式文件
        # 将json文件保存为.json格式文件
        with open('./data/num.json', 'w+') as file:
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



if __name__ == '__main__':
    path = './data/num.json'  # 输入文件夹地址
    with open(path, 'r+') as file:
        content = file.read()
    content = json.loads(content)  # 将json格式文件转化为python的字典文件
    num = content['num'] # 统计文件夹中的文件个数
    while num < 6000:
        run(150,num)
        with open(path, 'r+') as file:
            content = file.read()
        content = json.loads(content)  # 将json格式文件转化为python的字典文件
        num = content['num']  # 统计文件夹中的文件个数
