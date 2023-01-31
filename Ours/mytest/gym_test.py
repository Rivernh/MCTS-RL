import gym
import sys
sys.path.insert(0,r'/home/ylh/MCTS-RL/Ours')
from gym_carla.envs import carla_env
from Module.mcts_pure import MCTSPlayer as MCTS_Pure
import numpy as np
from utils_ import process

def Env_init():
    # parameters for the gym_carla environment
    params = {
        'number_of_vehicles': 100,
        'display_size': 256,  # screen size of bird-eye render
        'max_past_step': 1,  # the number of past steps to draw
        'dt': 0.1,  # time interval between two frames
        'discrete': False,  # whether to use discrete control space
        'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
        'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
        'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
        'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
        'ego_vehicle_filter': 'vehicle.lincoln*',  # filter for defining ego vehicle
        'port': 2000,  # connection port
        'town': 'Town03',  # which town to simulate
        'task_mode': 'random',  # mode of the task, [random, roundabout (only for Town03)]
        'max_time_episode': 1000,  # maximum timesteps per episode
        'max_waypt': 12,  # maximum number of waypoints
        'obs_range': 32,  # observation range (meter)
        'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
        'd_behind': 12,  # distance behind the ego vehicle (meter)
        'out_lane_thres': 2.0,  # threshold for out of lane
        'desired_speed': 8,  # desired speed (m/s)
        'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
        'display_route': True,  # whether to render the desired route
        'pixor_size': 64,  # size of the pixor labels
        'pixor': False,  # whether to output PIXOR observation
        'action_num':10,
        'availables':np.arange(-5,5,10 / 10)
    }
    env = carla_env.CarlaEnv(params)
    #env = gym.make('CarlaEnv-v0')
    env.reset()
    return env

def run():

    try:
        env = Env_init()
        #model_file = 'current_policy.model'

        # uncomment the following line to play with pure MCTS (it's much weaker even with a larger n_playout)
        mcts_player = MCTS_Pure(env, c_puct=5, n_playout=10000)
        
        done = False

        while not done:
            next_obs, reward, done, info = env.step(move)
            move = mcts_player.get_action()
            

    except KeyboardInterrupt:
        print('\n\rquit')

if __name__ == '__main__':
    process.start_process()
    env = Env_init()
    done = False

    for i in range(100):
        print("Episode {} start!".format(i))
        while not done:
            
            next_obs, reward, done, info = env.step([2, 0])
            print(next_obs.items())

        done = False