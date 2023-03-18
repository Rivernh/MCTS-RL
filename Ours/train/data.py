# -*- coding: utf-8 -*-
"""
collect train data

@author: Liuhan Yin
"""
from __future__ import print_function
import sys
sys.path.insert(0, r'/home/ylh/MCTS-RL/CARLA')
sys.path.insert(0, r'/home/ylh/MCTS-RL/Ours')
from Module.mcts_pure import MCTSPlayer as MCTS_Pure
import numpy as np
from gym_carla.envs import carla_env
from utils.process import start_process, kill_process
from kinematics.model import Env_model
import cv2
import json
import os
import time
def Env_init():
    # parameters for the gym_carla environment
    params = {
        'number_of_vehicles': 0,
        'display_size': 256,  # screen size of bird-eye render
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.1,  # time interval between two frames
        'discrete_acc': np.arange(-5,5,1),  # discrete value of accelerations
        'discrete_steer': np.arange(-5,5,1) / 10,  # discrete value of steering angles
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'opponent_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'port': 2000,  # connection port
        'town': 'Town03',  # which town to simulate
        'max_time_episode': 1000,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'obs_range': 32,  # observation range (meter)
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 10,  # desired speed (m/s)
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
        'display_route': True,  # whether to render the desired route
        'ego_transform': (2.2, 77.1, -88.1),
        'target_transform': (8.0, -54.4, -89.3),
        'opponent_transform': (59.9, -7.6, -179.1),
        'opponent_add': False
    }
    env = carla_env.CarlaEnv(params)
    env.reset()
    return env

def run():
    start_process(show=False)
    mcts_player = MCTS_Pure(c_puct=1, n_playout=500)
    do = True
    try:
        while do:
            path = '../data/img/'  # 输入文件夹地址
            files = os.listdir(path)  # 读入文件夹
            num = len(files)  # 统计文件夹中的文件个数
            print(f"png num:{num}")
            if num > 500:
                do = False
            done = False
            move = [0,0]
            car_env = Env_init()
            vehicle = []
            vehicle.append(car_env.ego)
            while not done:
                next_obs, reward, done, info = car_env.step(move)
                print(f"num:{num}")
                state = Env_model(vehicle)
                move = mcts_player.get_action(state)
                del state

                if car_env.time_step % 10 == 0:
                    #print(f"move:{move}")
                    # 保存观测信息
                    img = cv2.cvtColor(next_obs.front_view, cv2.COLOR_BGR2RGB)
                    cv2.imwrite("../data/img/" + f'{num}.png', img)
                    time.sleep(0.1)

                    state = {"ve": next_obs.speed, "gama_e": 1.0, "vo": 0, "gama_o": 0.1, "dis": 100, "theta": 0}
                    dict_json = json.dumps(state)  # 转化为json格式文件
                    # 将json文件保存为.json格式文件
                    with open('../data/obs/' + f'{num}.json', 'w+') as file:
                        file.write(dict_json)
                    num += 1
                if num > 10:
                    do = False
                    break
            print("done!")

    except KeyboardInterrupt:
        print('\n\rquit')
    finally:
        car_env._set_synchronous_mode(False)
        kill_process()


if __name__ == '__main__':
    run()


