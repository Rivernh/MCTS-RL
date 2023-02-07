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
import gym
from gym_carla.envs import carla_env
from tqdm import tqdm
from utils.process import start_process,kill_process
import itertools
from dynamics.env import Dynamics_Env

def Env_init():
    # parameters for the gym_carla environment
    params = {
        'number_of_vehicles': 0,
        'display_size': 256,  # screen size of bird-eye render
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.05,  # time interval between two frames
        'discrete_acc': np.arange(-5,5,1),  # discrete value of accelerations
        'discrete_steer': np.arange(-5,5,1) / 10,  # discrete value of steering angles
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
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
    }
    env = carla_env.CarlaEnv(params)
    env.reset()
    return env

def test():
    env = Env_init()
    bp_lib = env.world.get_map().get_spawn_points()
    print(bp_lib)



def run():
    #model_file = 'current_policy.model'
    car_env = Env_init()
    dynamics_env = Dynamics_Env()
    try:


        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        mcts_player = MCTS_Pure(c_puct=5, n_playout=5000)
        done = False
        move = 0.0
        while not done:
            next_obs, reward, done, info = car_env.step(move)
            done = False
            if car_env.time_step % 10 == 0:
                move = mcts_player.get_action(dynamics_env, car_env.ego, car_env.waypoints)
                print(f"move:{move}")
        print("done!")
            
    except KeyboardInterrupt:
        print('\n\rquit')
    finally:
        car_env._set_synchronous_mode(False)
        kill_process()



if __name__ == '__main__':
    start_process(show=False)

    run()
