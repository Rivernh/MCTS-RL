import numpy as np
import math
import itertools
import copy
from gym_carla.envs.misc import *
import queue


class Dynamics_Env():
    """a kinematics env of carla"""
    def __int__(self):
        self.origin_pos = None
        self.availables = list(itertools.product(np.array([0.5, 0.3, 0, -0.3]), np.array([0, -0.2, 0.2, -0.4, 0.4])))

    def reset(self, vehicle, waypoints):
        self.availables = np.array([2,1,0,-1,-2])
        self.delta_t = 0.5
        self.out_lane_thres = 2
        self.current_pos = np.zeros(3)
        self.current_speed = np.zeros(2)
        self.step = 0
        self.max_step = 20
        self.success = False
        self.end = False
        self.hit = False
        self.distance_followed = 0
        self.outlane = False
        self.waypoints = None
        self.pos = 0
        self.wpt_len = len(waypoints)
        self.waypoints = copy.deepcopy(waypoints)
        speed = vehicle.get_velocity()
        self.current_speed[0] = speed.x
        self.current_speed[1] = speed.y
        self.transform = vehicle.get_transform()

        self.origin_pos = get_info(vehicle)
        self.current_pos[0] = self.origin_pos[0]
        self.current_pos[1] = self.origin_pos[1]
        self.current_pos[2] = self.origin_pos[2]

        self.max_speed = 10
        self.speed = 0
        self.acc = 0

    """
    将指令转换为速度信息
    每个油门指令有一个对应的最大速度，若为达到则加速，若到达，则保持
    若油门为0或负，则有减速，且最多减速为0，不能变向加速
    
    近似认为动作执行后，速度方向与汽车方向一致
    
    """
    def take_action(self, action):
        temp_target = self.waypoints[1]  #
        yaw = self.current_pos[2]
        car_dir = np.array([math.cos(yaw), math.sin(yaw)])
        s_dir = np.array([temp_target[0] - self.current_pos[0], temp_target[1] - self.current_pos[1]])
        cos_theta = np.dot(car_dir, s_dir) / (np.linalg.norm(car_dir) * np.linalg.norm(s_dir))
        left_right = abs(np.cross(car_dir, s_dir)) / np.cross(car_dir, s_dir)
        angle = np.arccos(cos_theta) * left_right

        steer = angle * 2
        acc = action
        self.acc = acc

        v = math.sqrt(self.current_speed[0]**2 + self.current_speed[1]**2)
        v_after = acc * self.delta_t + v
        if acc > 0:
            if v_after >= self.max_speed:
                v_after = self.max_speed
        else:
            if v_after <= 0:
                v_after = 0

        self.current_speed[0] = v_after * math.cos(yaw+steer)
        self.current_speed[1] = v_after * math.sin(yaw+steer)
        #TODO 计算车辆的角度
        v = math.sqrt(self.current_speed[0] ** 2 + self.current_speed[1] ** 2)
        self.speed = v
        if self.current_speed[0]!=0:
            yaw = math.atan(self.current_speed[1]/self.current_speed[0])
            if v > 0:
                if yaw > 0:
                    if self.current_speed[0] > 0:
                        self.current_pos[2] = yaw
                    elif self.current_speed[0] < 0:
                        self.current_pos[2] = yaw - math.pi
                else:
                    if self.current_speed[0] > 0:
                        self.current_pos[2] = yaw
                    elif self.current_speed[0] < 0:
                        self.current_pos[2] = yaw + math.pi

    """根据动作执行一次"""
    def do_move(self, action):
        self.take_action(action)
        self.step += 1
        delta_x = self.delta_t * self.current_speed[0]
        delta_y = self.delta_t * self.current_speed[1]
        self.current_pos[0] += delta_x
        self.current_pos[1] += delta_y

    """do a move keep speed"""
    def next(self):
        self.step += 1
        delta_x = self.delta_t * self.current_speed[0]
        delta_y = self.delta_t * self.current_speed[1]
        self.current_pos[0] += delta_x
        self.current_pos[1] += delta_y
        if self.arrive():
            return True, True
        if self.hist():
            return True, False
        if self.out_lane():
            return True, False
        if self.step > self.max_step:
            return True, False
        return False, False

    """run until the episode end """
    def run_until_end(self):
        while not self.end:
            self.end, self.success = self.next()
        reward = self.get_reward()
        return reward

    """get the episode reward"""
    def get_reward(self):
        speed_reward = 1 - np.exp(-(self.speed**2))
        acc_reward = np.exp(-(self.acc**2))
        #print(f"s rew:{speed_reward}    a rew:{acc_reward}")

        reward = speed_reward + 0.5*acc_reward
        return reward

    """hit other car"""
    #TODO add hit check
    def hist(self):
        self.hit = False
        return self.hit

    """check if arrive the end"""
    def arrive(self):
        return False

    def out_lane(self):
        dis, w = get_lane_dis(self.waypoints, self.current_pos[0], self.current_pos[1])
        if abs(dis) > self.out_lane_thres:
            self.outlane = True
            return True
       # print(f"dis:{dis}")
     #   print(f"step:{self.step}")
        return False

    """check if end"""
    def terminal(self):
        if self.hist():
            return True
        if self.arrive():
            return True
        #if self.out_lane():
        #    return True
        if self.step > self.max_step:
            return True
        return False










