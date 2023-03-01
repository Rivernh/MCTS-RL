# -*- coding: utf-8 -*-
"""
MCTS pure for autonomous driving

@author: Liuhan Yin
"""

from __future__ import print_function
import sys
sys.path.insert(0,r'/home/ylh/MCTS-RL/CARLA')
sys.path.insert(0,r'/home/ylh/MCTS-RL/Ours')
from Module.mcts_pure import MCTSPlayer as MCTS_Pure
import numpy as np
from gym_carla.envs import carla_env
from utils.process import start_process,kill_process
from kinematics.model import Env_model
import matplotlib.pyplot as plt


def Env_init():
    # parameters for the gym_carla environment
    params = {
        'number_of_vehicles': 0,
        'display_size': 256,  # screen size of bird-eye render
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.05,  # time interval between two frames
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
        'display_route': False,  # whether to render the desired route
        'ego_transform': (43.8, -7.8, -179.0),
        'barrier_transform': (17.9, 8.7, -86.1),
        'target_transform': (4, -45, -89.3),
    }
    env = carla_env.CarlaEnv(params)
    env.reset()
    return env

def run():
    start_process(show=True)
    car_env = Env_init() #carla env
    vehicle = []
    wpt = []
    ap_vehicle = []
    vehicle.append(car_env.ego) #控制的智能体
    ap_vehicle.append(car_env.barrier) #自动驾驶的智能体
    time = []
    v = []
    barrier_v = []
    target_v = []
    try:
        mcts_player = MCTS_Pure(c_puct=1, n_playout=200)
        done = False
        move = (0,0)
        while not done:
            next_obs, reward, done, info = car_env.step(move)
            wpt.clear()
            wpt.append(car_env.waypoints[0:3])
            if car_env.time_step % 2 == 0:
                state = Env_model(vehicle, wpt,ap_vehicle) #动力学仿真环境
                move = mcts_player.get_action(state)
                del state
                print(f"move:{move}")

                # 绘图数据
                time.append(car_env.time_step*0.05)
                v.append(car_env.v_now)
                barrier_v.append(car_env.barrier_v)
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

        plt.plot(time,v,color='#1E90FF',linewidth=1.0,linestyle='-',marker='o',label='ego agent')
        plt.plot(time, barrier_v, color='#FFA500', linewidth=1.0, linestyle='-', marker='D', label='barrier agent')
        plt.plot(time, np.array(target_v)[:,0], color='#6A5ACD', linewidth=1.0, linestyle=':', marker='s', label='target speed')
        #plt.plot(time, np.array(target_v)[:,1], color='#708090', linewidth=1.0, linestyle=':', marker='s', label='target steer')
        plt.legend(loc='best')  # 图例展示位置，数字代表第几象限3
        plt.show()

if __name__ == '__main__':
    run()
