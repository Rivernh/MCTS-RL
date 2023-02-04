import numpy as np
import math
import itertools
import copy
from gym_carla.envs.misc import *

class Dynamics_Env():
    """a dynamics env of carla"""
    def __int__(self):
        self.current_cmd = None
        self.distance_followed = 0
        self.delta_t = 0.5
        self.step = 0
        self.success = False
        self.end = False
        self.hit = False
        self.action_num = 25
        self.outlane = False
        self.waypoints = None
        self.pos = 0
        self.out_lane_thres = 0.2
        self.origin_pos = None

    def reset(self, vehicle, waypoints):
        self.availables = list(itertools.product(np.array([0.6,0.3,0,-0.3]), np.array([0,-0.2,0.2,-0.4,0.4])))
        self.delta_t = 0.5
        self.out_lane_thres = 0.3
        self.current_pos = np.zeros(3)
        self.current_speed = np.zeros(2)
        self.step = 0
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

        self.origin_pos = get_info(vehicle)
        self.current_pos[0] = self.origin_pos[0]
        self.current_pos[1] = self.origin_pos[1]
        self.current_pos[2] = self.origin_pos[2]

    def take_action(self, action):
        acc = action[0]
        steer = action[1]
        temp = np.zeros(2)
        theta = self.current_pos[2]

        delta_v = acc * self.delta_t

        temp[0] = self.current_speed[0] + delta_v * math.sin(theta - steer)
        temp[1] = self.current_speed[1] + delta_v * math.sin(theta - steer)

        if acc > 0:
            self.current_speed[0] = temp[0]
            self.current_speed[1] = temp[1]
        else:
            if (temp[0]**2 + temp[1]**2) < (self.current_speed[0]**2 + self.current_speed[1]**2):
                self.current_speed[0] = temp[0]
                self.current_speed[1] = temp[1]

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
        if self.step > 1000:
            return True, False
        return False, False

    def run_until_end(self):
        while not self.end:
            self.end, self.success = self.next()
        reward = self.get_reward()
        return reward

    def get_reward(self):
        """速度奖励"""
        speed = math.sqrt(np.sum(self.current_speed[0] ** 2))
        speed_reward = 10 * speed
        """距离奖励"""
        distance_reward = 1 * self.pos
        """完成奖励"""
        if self.success:
            finished_reward = 1
        else:
            finished_reward = 0
        """碰撞奖励"""
        if self.hit:
            crash_reward = -1
        else:
            crash_reward = 0
        """偏离路径太大奖励"""
        if self.outlane:
            out_reward = -1
        else:
            out_reward = 0
        """时间奖励"""
        step_reward = -0.01 * self.step
        reward = speed_reward + distance_reward + finished_reward + crash_reward + step_reward + out_reward
        return reward

    """hit other car"""
    #TODO add hit check
    def hist(self):
        self.hit = False
        return self.hit

    def arrive(self):
        #self.pos_check()
        if self.pos >= self.wpt_len - 2:
            return True
        return

    def out_lane(self):
        dis, _ = get_lane_dis(self.waypoints, self.current_pos[0], self.current_pos[1])
        if abs(dis) > self.out_lane_thres:
            self.outlane = True
            return True
       # print(f"dis:{dis}")
     #   print(f"step:{self.step}")
        return False

    def pos_check(self):
        waypoint_before = self.waypoints[0]
        for index in np.arange(self.wpt_len-2):
            waypoint_mid = self.waypoints[index+1]
            waypoint_after = self.waypoints[index+2]
            k0 = (waypoint_mid[1] - waypoint_before[1]) / (waypoint_mid[0] - waypoint_before[0])
            k0 = -1 / k0
            value0 = k0 * (self.current_pos[0] - (waypoint_mid[0] + waypoint_before[0]) / 2) - self.current_pos[1] + (waypoint_mid[1] + waypoint_before[1]) / 2

            k1 = (waypoint_after[1] - waypoint_mid[1]) / (waypoint_after[0] - waypoint_mid[0])
            k1 = -1 / k1
            value1 = k1 * (self.current_pos[0] - (waypoint_after[0] + waypoint_mid[0]) / 2) - self.current_pos[1] + (waypoint_after[1] + waypoint_mid[1]) / 2

            waypoint_before = waypoint_mid

            if value1 * value0 <= 0:
                self.pos = index + 1
                break
        else:
            self.pos = 0
        print(f"self.pos:{self.pos}")
        return self.pos

    def terminal(self):
        if self.hist():
            return True
        if self.arrive():
            return True
        if self.out_lane():
            return True
        if self.step > 1000:
            return True
        return False










