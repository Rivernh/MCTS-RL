# -*- coding: utf-8 -*-
"""
MCTS for autonomous driving

@author: Liuhan Yin
"""

from __future__ import print_function
import sys
sys.path.insert(0,r'/home/ylh/MCTS-RL/CARLA')
sys.path.insert(0,r'/home/ylh/MCTS-RL/Ours')
from Module.mcts.mcts_withnet import MCTSPlayer as MCTS_Pure
import numpy as np
from gym_carla.envs import carla_env
from utils.process import start_process,kill_process
from kinematics.model import Env_model
import matplotlib.pyplot as plt
import time as t

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
        'ego_transform': [(149,-25.2,90),(165.5,-2.3,180.2)],
        'target_transform': (80,-5.3,180.2),
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
    vehicle = car_env.egos #控制的智能体
    #vehicle.append(car_env.barrier) #控制的智能体
    time = []
    v = []
    target_v = []
    try:
        car1 = MCTS_Pure(c_puct=5, n_playout=300)
        car2 = MCTS_Pure(c_puct=5, n_playout=300)
        done = False
        car1_move = (0,0)
        car2_move = (0, 0)
        move = [car1_move,car2_move]
        while not done:
            next_obs, reward, done, info = car_env.step(move)
            done = False
            wpt.clear()
            wpt_temp = car_env.waypoints_all
            if len(wpt_temp[0]) < 3 or len(wpt_temp[1]) < 3:
                break
            for i in range(len(wpt_temp)):
                wpt.append(wpt_temp[i][0:3])
            if car_env.time_step % 2 == 0:
                state = Env_model(vehicle, wpt,ap_vehicle,dt=0.1) #动力学仿真环境


                car2_move = car2.get_action(state,1)
                car1_move = car1.get_action(state,0)
                move = [car1_move, car2_move]
                del state
                print(f"move:{move}")

            # 绘图数据
            time.append(car_env.time_step*0.05)
            v.append(car_env.v_now)
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

        plt.plot(time,v[:,0],color='#1E90FF',linewidth=1.0,linestyle='-',marker='o',label='ego agent1')
        plt.plot(time, v[:,1], color='#6A5ACD', linewidth=1.0, linestyle='-', marker='o', label='ego agent2')
        #plt.plot(time, np.array(target_v)[:,0], color='#6A5ACD', linewidth=1.0, linestyle=':', marker='s', label='target speed')
        #plt.plot(time, np.array(target_v)[:,1], color='#708090', linewidth=1.0, linestyle=':', marker='s', label='target steer')
        plt.legend(loc='best')  # 图例展示位置，数字代表第几象限3
        plt.show()

if __name__ == '__main__':
    run()
