import glob
import os
import sys
import random
import os
from queue import Queue
from queue import Empty

import carla
from carla import Transform, Location, Rotation
import numpy as np
import gym
from static_env import StaticEnv
#可能的动作为加速度标量的变化，智能体在原本驾驶路线轨迹上行驶，算法输出叠加一个速度标量的变化控制
from utils_ import action_available,action_index,action_num
from utils_ import Mypara,Myagents

from simulator import World

class CarlaEnv(gym.Env, StaticEnv):
    """
    简单的模拟环境，目标是从其实模拟位置进行交互动作完成交互任务。
    奖励被定义速度和加速度的函数，以及驾驶风格影响的奖励计算。
    """

    n_actions = action_num

    def __init__(self):
        self.agent_num = 2
        self.ep_length = 15

        self.init_pos = [(6, 0),(6, 6)]
        self.pos = self.init_pos
        self.step_idx = 0

        #carla init
        self.actor_list = []
        self.sensor_list = []

        # 连接Carla服务器
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        sim_world = self.client.get_world()
        self.world = World(sim_world,None,)
        self.client.load_world('Town05')

        self.sensor_queue = Queue()
        #获取Carla中所有蓝图
        self.blueprint_library = sim_world.get_blueprint_library()
        #获取车联蓝图
        self.vehicle_blueprints = self.blueprint_library.filter('*vehicle*')
        #随机选择其中一个车辆蓝图
        self.ego_vehicle_bp = random.choice(self.vehicle_blueprints)

        #获取Carla中所有出生点
        self.birth_point = sim_world.get_map().get_spawn_points()
        #随机选择其中一个出生点
        self.transform = random.choice(self.birth_point)
        #创建车辆
        self.ego_vehicle = sim_world.spawn_actor(self.ego_vehicle_bp, self.transform)
        self.ego_vehicle.set_autopilot(True)
        self.actor_list.append(self.ego_vehicle)

        #设置观察者视角
        self.spectator = sim_world.get_spectator()
        transform = self.ego_vehicle.get_transform()
        self.spectator.set_transform(carla.Transform(transform.location + carla.Location(z=20),
                                                    carla.Rotation(pitch=-90)))

        #self.ego_vehicle.set_autopilot(True)
        self.ego_vehicle.apply_control(carla.VehicleControl(throttle = 1.0, steer = 0.0))

        # 创建相机传感器
        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = sim_world.spawn_actor(self.camera_bp, self.camera_transform, attach_to=self.ego_vehicle)
        # 注册回调函数
        self.camera.listen(lambda image: sensor_callback(image, self.sensor_queue, "camera"))
        self.sensor_list.append(self.camera)

    def reset(self):
        self.pos = self.init_pos
        self.step_idx = 0
        state = self.pos[0]*self.shape[0] + self.pos[1]
        return state, 0, False, None

    def step(self, action):
        self.step_idx += 1
        reward = 0   # -0.5 for encouraging speed
        state = 0
        done = self.pos == (0, 6) or self.step_idx == self.ep_length
        return state, reward, done, None

    def render(self):
        pass

    def end(self):
        print('destroying actors')
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        for sensor in self.sensor_list:
            sensor.destroy()
        for actor in self.actor_list:
            actor.destroy()
        print('done.')

    @staticmethod
    def next_state(state, action, shape=(7, 7)):
        pos = np.unravel_index(state, shape)
        return pos[0] * shape[0] + pos[1]

    @staticmethod
    def is_done_state(state, step_idx, shape=(7, 7)):
        return np.unravel_index(state, shape) == (0, 6) or step_idx >= 15

    @staticmethod
    def initial_state(shape=(7, 7)):
        return (shape[0]-1) * shape[0]

    @staticmethod
    def get_obs_for_states(states):
        return np.array(states)

    @staticmethod
    def get_return(state, step_idx, shape=(7, 7)):
        row, col = np.unravel_index(state, shape)
        return CarlaEnv.altitudes[row][col] - step_idx*0.5

    @staticmethod
    def _limit_coordinates(coord, shape):
        """
        Prevent the agent from falling out of the grid world.
        """
        coord = list(coord)
        coord[0] = min(coord[0], shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return tuple(coord)

if __name__ == '__main__':
    print(action_available)
