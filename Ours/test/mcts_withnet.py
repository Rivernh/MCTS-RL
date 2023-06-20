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

def policy_value_fn(TreeEnv):
    """a function that takes in a obs and outputs a list of (action, probability)
    tuples and a score for the obs"""
    # return uniform probabilities and 0 score for pure MCTS
    availables = np.arange(25)
    action_probs = np.ones(len(availables))/len(availables)
    return zip(availables, action_probs), 0

def run():
    start_process(show=True)
    car_env = Env_init()  # carla env
    vehicle = []
    wpt = []
    ap_vehicle = []
    vehicle = car_env.egos  # 控制的智能体
    # vehicle.append(car_env.barrier) #控制的智能体
    time = []
    d = [[0,0]]
    v = []
    target_v = []
    try:
        #best_policy = PolicyValueNet(3,64,1,2,model_file=r'/home/ylh/MCTS-RL/Ours/model_save/init.pt',use_gpu=True)
        #car = MCTS(best_policy.policy_value_fn,c_puct=0.2, n_playout=50)
        done = False
        car1_move = (0, 0)
        car2_move = (0, 0)
        move = [car1_move, car2_move]
        while not done:
            next_obs, _, done, _ = car_env.step(move)
            wpt.clear()
            wpt_temp = car_env.waypoints_all
            if len(wpt_temp[0]) < 8 or len(wpt_temp[1]) < 8:
                break
            for i in range(len(wpt_temp)):
                wpt.append(wpt_temp[i][2:8])
            if car_env.time_step % 2 == 0:
                state = Env_model(vehicle, wpt, ap_vehicle, dt=0.1)  # 动力学仿真环境

                #car1_move,p1 = car.get_action(state, 0)
                #car2_move,p2 = car.get_action(state, 1)
               # move = [car1_move, car2_move]
                print(f"move:{move}")
                del state
            # 绘图数据
            time.append(car_env.time_step * 0.05)
            v.append(car_env.v_now)
            dis = [d[len(d)-1][0] + car_env.v_now[0] * 0.05,d[len(d)-1][1] + car_env.v_now[1] * 0.05]
            d.append(dis)
            target_v.append(move)
        print("done!")

    except KeyboardInterrupt:
        print('\n\rquit')
    finally:
        car_env._set_synchronous_mode(False)
        kill_process()
        plt.title('speed curve', fontsize=15)  # 标题，并设定字号大小
        plt.xlabel(u'time(s)', fontsize=10)  # 设置x轴，并设定字号大小
        plt.ylabel(u'speed(m/s)', fontsize=10)  # 设置y轴，并设定字号大小
        v = np.array(v)

        plt.plot(time, v[:, 0], color='#6495ED', linewidth=2.0, linestyle='-', marker='o', label='Agent1')
        plt.plot(time, v[:, 1], color='#FF8C00', linewidth=2.0, linestyle='-', marker='s', label='Agent2')
        # plt.plot(time, np.array(target_v)[:,0], color='#6A5ACD', linewidth=1.0, linestyle=':', marker='s', label='target speed')
        #plt.plot(time, np.array(target_steer)[:,0], color='#6A5ACD', linewidth=1.0, linestyle=':', marker='s', label='target steer')
        #plt.plot(time, np.array(target_steer)[:, 1], color='#708090', linewidth=1.0, linestyle=':', marker='s',
        #         label='target steer')
        plt.legend(loc='best')  # 图例展示位置，数字代表第几象限3
        plt.show()

        d.pop(0)
        d = np.array(d)

        plt.title('distance curve', fontsize=15)  # 标题，并设定字号大小
        plt.xlabel(u'time(s)', fontsize=10)  # 设置x轴，并设定字号大小
        plt.ylabel(u'distance(m)', fontsize=10)  # 设置y轴，并设定字号大小

        plt.plot(time, d[:, 0], color='#6495ED', linewidth=2.0, linestyle='-', marker='o', label='Agent1')
        plt.plot(time, d[:, 1], color='#FF8C00', linewidth=2.0, linestyle='-', marker='s', label='Agent2')
        # plt.plot(time, np.array(target_v)[:,0], color='#6A5ACD', linewidth=1.0, linestyle=':', marker='s', label='target speed')
        #plt.plot(time, np.array(target_steer)[:,0], color='#6A5ACD', linewidth=1.0, linestyle=':', marker='s', label='target steer')
        #plt.plot(time, np.array(target_steer)[:, 1], color='#708090', linewidth=1.0, linestyle=':', marker='s',
        #         label='target steer')
        plt.legend(loc='best')  # 图例展示位置，数字代表第几象限3
        plt.show()



if __name__ == '__main__':
    run()
