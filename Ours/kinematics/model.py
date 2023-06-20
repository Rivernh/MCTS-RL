import numpy as np
import math
import matplotlib.pyplot as plt
import time
import itertools
from scipy.stats import norm

def my_get_info(vehicle):
    trans = vehicle.get_transform()
    x = trans.location.x
    y = trans.location.y
    yaw = trans.rotation.yaw / 180 * np.pi
    bb = vehicle.bounding_box
    l = bb.extent.x
    w = bb.extent.y
    wheels = vehicle.get_physics_control().wheels
    front = [(wheels[0].position.x + wheels[1].position.x) / 2, (wheels[0].position.y + wheels[1].position.y) / 2]
    back = [(wheels[2].position.x + wheels[2].position.x) / 2, (wheels[3].position.y + wheels[3].position.y) / 2]
    wheel_base = math.sqrt((front[0]-back[0])**2 + (front[1]-back[1])**2) / 100
    info = [x, y, yaw, wheel_base, l, w]
    return info

def my_caculate(a, b, p):  # 判断p在ab上的投影是否在线段上 0不在 1在 2在端点
    a = np.array(a)
    b = np.array(b)
    p = np.array(p)
    v = b - a
    u = p - a
    count = 0
    if ((np.dot(v, u) / np.dot(v, v)) >= 0) and ((np.dot(v, u) / np.dot(v, v)) <= 1):
        return True
    return False

def overlap(box1,box2):#TODO 加入已知矩形四个顶点，检测是否碰撞
    crash = True
    #box1
    for i in range(4):
        a = box1[i]
        b = box1[(i + 1) % 4]
        cnt = 0
        for point in box2:
            if my_caculate(a,b,point):
                break
            cnt += 1
        if cnt == 4:
            crash = False
            return crash
    #box2
    for i in range(4):
        a = box2[i]
        b = box2[(i + 1) % 4]
        cnt = 0
        for point in box1:
            if my_caculate(a,b,point):
                break
            cnt += 1
        if cnt == 4:
            crash = False
            return crash
    return crash

def inside_range(p,wpt,threlf=0.5,thrert=0.5):#TODO 加入在规划轨迹一定范围内的的检测，便于后续加入steer维度的动作
    #判断当前坐标是否在一个矩形区域内
    l = 20
    #计算矩形的顶点
    a = wpt[0][0:2]
    b = wpt[1][0:2]
    vec = b - a
    point = (b + a) / 2
    sin = vec[1] / np.linalg.norm(vec)
    cos = vec[0] / np.linalg.norm(vec)
    left_bottom = [-l * cos + threlf * sin + point[0],
                   -l * sin - threlf * cos + point[1]]
    right_bottom = [l * cos + thrert * sin + point[0],
                    l * sin - thrert * cos + point[1]]
    right_up = [l * cos - thrert * sin + point[0],
                l * sin + thrert * cos + point[1]]
    left_up = [-l * cos - threlf * sin + point[0],
               -l * sin + threlf * cos + point[1]]
    box = [left_bottom, right_bottom, right_up, left_up]

    p = np.array(p)
    for i in range(4):
        a = box[i]
        b = box[(i + 1) % 4]
        if not my_caculate(a, b, p):
            return False
    return True

class UGV_model:
    def __init__(self, info, T):  # L:wheel base
        self.x = info[0] # X
        self.y = info[1]  # Y
        self.theta = info[2]  # headding
        self.wb = info[3]  # wheel base
        self.l = info[4]
        self.w = info[5]

        self.dt = T  # decision time periodic

    def update(self, vt, deltat):  # update ugv's obs
        self.v = vt

        dx = self.v * np.cos(self.theta)
        dy = self.v * np.sin(self.theta)
        dtheta = self.v * np.tan(deltat) / self.wb

        self.x += dx * self.dt
        self.y += dy * self.dt
        self.theta += dtheta * self.dt

        #更新box 四个顶点
        left_bottom = [-self.l*math.cos(self.theta) + self.w*math.sin(self.theta) + self.x,
                       -self.l*math.sin(self.theta) - self.w*math.cos(self.theta) + self.y]
        right_bottom = [self.l*math.cos(self.theta) + self.w*math.sin(self.theta) + self.x,
                        self.l*math.sin(self.theta) - self.w*math.cos(self.theta) + self.y]
        right_up = [self.l*math.cos(self.theta) - self.w*math.sin(self.theta) + self.x,
                    self.l*math.sin(self.theta) + self.w*math.cos(self.theta) + self.y]
        left_up = [-self.l*math.cos(self.theta) - self.w*math.sin(self.theta) + self.x,
                   -self.l*math.sin(self.theta) + self.w*math.cos(self.theta) + self.y]
        self.box = [left_bottom,right_bottom,right_up,left_up]


    def plot_duration(self):
        plt.scatter(self.x, self.y, color='r')
        #plt.axis([-5, 100, -6, 6])
        plt.pause(0.008)

class ap_model:
    def __init__(self, info, v, dt=0.2):  # L:wheel base
        self.x = info[0]  # X
        self.y = info[1]  # Y
        self.theta = info[2]  # headding
        self.wb = info[3]  # wheel base
        self.l = info[4] + 0.5
        self.w = info[5] + 0.5
        self.dt = dt
        self.v = v

    def update(self):  # update ugv's obs
        dx = self.v * np.cos(self.theta)
        dy = self.v * np.sin(self.theta)

        self.x += dx * self.dt
        self.y += dy * self.dt

        #更新box 四个顶点
        left_bottom = [-self.l*math.cos(self.theta) + self.w*math.sin(self.theta) + self.x,
                       -self.l*math.sin(self.theta) - self.w*math.cos(self.theta) + self.y]
        right_bottom = [self.l*math.cos(self.theta) + self.w*math.sin(self.theta) + self.x,
                        self.l*math.sin(self.theta) - self.w*math.cos(self.theta) + self.y]
        right_up = [self.l*math.cos(self.theta) - self.w*math.sin(self.theta) + self.x,
                    self.l*math.sin(self.theta) + self.w*math.cos(self.theta) + self.y]
        left_up = [-self.l*math.cos(self.theta) - self.w*math.sin(self.theta) + self.x,
                   -self.l*math.sin(self.theta) + self.w*math.cos(self.theta) + self.y]
        self.box = [left_bottom,right_bottom,right_up,left_up]

class Env_model(object):
    def __init__(self, vehicle, wpt, ap_vehicle, dt = 0.2):
        #self.availables = list(itertools.product(np.array([0,3,6,9,12]), np.array([0,-0.2,-0.4,0.2,0.4])))
        #self.availables = np.array([0,2.5,5,7.5,10])
        self.availables = list(np.arange(21))
        self.availables_len = 21
        self.dt = dt
        self.vehicle = vehicle
        self.ap_vehicle = ap_vehicle
        self.vehicle_num = len(self.vehicle)
        self.AP_num = len(self.ap_vehicle)
        self.total = self.vehicle_num + self.AP_num
        self.crash = np.zeros(self.vehicle_num)
        self.wpt = wpt
        self.UGV_info = []
        self.AP_info = []
        self.speed_init = []
        self.speed_temp = []
        self.acc_init = [0]
        #print(self.vehicle_num)
        while len(self.UGV_info) < self.vehicle_num:
            v = self.vehicle[len(self.UGV_info)].get_velocity()
            v = math.sqrt(v.x ** 2 + v.y ** 2)
            self.speed_init.append(v)
            self.speed_temp.append(v)
            info = my_get_info(self.vehicle[len(self.UGV_info)])
            self.UGV_info.append(info)

        while len(self.AP_info) < self.AP_num:
            v = self.ap_vehicle[len(self.AP_info)].get_velocity()
            v = math.sqrt(v.x ** 2 + v.y ** 2)
            info = my_get_info(self.ap_vehicle[len(self.AP_info)])
            self.AP_info.append([info,v])

    def reset(self):
        self.step = 0
        self.end = False
        self.UGV = []
        self.AP = []
        self.speed = []
        self.acc = [0]
        self.steer = [0, 0]
        while len(self.UGV) < self.vehicle_num:
            self.speed.append(self.speed_init[len(self.UGV)])
            self.UGV.append(UGV_model(self.UGV_info[len(self.UGV)], self.dt))
            #if len(self.speed):
            #    print("successfully add speed!")

        while len(self.AP) < self.AP_num:
            self.AP.append(ap_model(self.AP_info[len(self.AP)][0],self.AP_info[len(self.AP)][1]))
            #if len(self.speed):
            #    print("successfully add speed!")

    # do a move according to the action for one car
    def do_move(self, action, rank):
        v = action[0]
        steer = action[1]
        self.steer[rank] = steer
        #self.acc[rank] = abs(v-self.speed[rank])
        self.UGV[rank].update(v, steer)
        self.speed[rank] = v
        #print("move done!")
        return

    #
    def change_init(self,action,rank):
        v = action[0]
        steer = action[1]
        theta = self.UGV_info[rank][2]

        dx = v * np.cos(theta) * self.dt
        dy = v * np.sin(theta) * self.dt
        dtheta = v * np.tan(steer) / self.UGV_info[rank][3] * self.dt

        self.speed_init[rank] = v
        self.UGV_info[rank][0] += dx
        self.UGV_info[rank][1] += dy
        self.UGV_info[rank][2] += dtheta


    # run an episode until end keeping the speed after the action for one agent
    def run_episode(self,rank):
        # game intention
        num = self.vehicle_num
        while num > 0:
            num = num - 1
            if num > rank:
                self.speed[num] /= 1#2 1.5
            elif num < rank:
                self.speed[num] *= 1

        while not self.end:
            for i in range(self.vehicle_num):
                self.UGV[i].update(self.speed[i], 0)
            for i in range(self.AP_num):
                self.AP[i].update()
            self.end = self.terminal(rank)
            self.step += 1
        return self.get_reward(rank)

    # get the reward of the action node
    def get_reward(self,rank):
        reward = []
        for i in range(self.vehicle_num):
            speed_reward = (1 - np.exp(-0.025*(self.speed[i] ** 2)))*1
            #acc_reward = np.exp1(-0.01*self.dt*(self.acc[i] ** 2))
            #acc_reward = self.acc[i] / 10
            task_reward = self.task(i)
            #offset_reward = -abs(self.offset[i][1])*0.04 + -abs(self.offset[i][0])*0.01 #转盘
            offset_reward = -abs(self.offset[i]) * 0.1 #右转 0.1 0.2
            if abs(self.offset[i]) > (1.57079):
                add_rew = -1
            else:
                add_rew = 0
            if self.steer[rank] < 0:
                turn_reward = -abs(self.steer[rank]) * 0.2
            else:
                turn_reward = -abs(self.steer[rank]) * 0.2

            rew = speed_reward + task_reward + offset_reward  + turn_reward
            prob = self.prob_caculate(i)
            rew = rew * prob
            reward.append(rew)
            #print(reward)

        return reward[rank] * 0.1

    # check if the episode is end and update the UGV obs flag in the env
    def terminal(self,rank):
        self.crash_check()
        self.off_track()
        if self.step >= 5:
            return True
        if self.crash[rank]:
            return True
        #if not self.judge[0]:
        #    return True
        return False

    def task(self,rank):
        rew = 0
        if self.crash[rank]:
            rew += -0.5 * (10 - self.step) * 10
        if self.speed[rank] == 0:
            rew += -0.5 * 0.1

        #if not self.judge[0]:
        #    rew += -0.001 * (10 - self.step)
        #    print("out")
        return rew

    def crash_check(self):
        for i in range(self.vehicle_num):
            crash = False
            for j in range(self.total-1):
                if j + 1 >= self.vehicle_num:
                    crash = overlap(self.UGV[i].box,self.AP[j+1-self.vehicle_num].box)#与周围车辆非控制的检测
                    #print(f"dis:{dis} thre:{thre}")
                else:
                    if j<i:
                        crash = overlap(self.UGV[i].box, self.UGV[j].box)#与其他UGV检测
                    else:
                        crash = overlap(self.UGV[i].box, self.UGV[j+1].box)  # 与其他UGV检测
                if crash:
                    break
            i#f not self.crash[i]:
            self.crash[i] = crash

    def off_track(self):
        """
        计算每个智能体偏离预定轨迹的距离，来控制循迹
        """
        self.offset = []
        for i in range(self.vehicle_num):
            wpt = self.wpt[i]
            temp_target = wpt[3]  #
            yaw = self.UGV[i].theta
            car_dir = np.array([math.cos(yaw), math.sin(yaw)])
            s_dir = np.array([temp_target[0] - self.UGV[i].x, temp_target[1] - self.UGV[i].y])
            cos_theta = np.dot(car_dir, s_dir) / (np.linalg.norm(car_dir) * np.linalg.norm(s_dir))
            left_right = (abs(np.cross(car_dir, s_dir)) + 1e-10) / (np.cross(car_dir, s_dir) + 1e-10)
            if abs(cos_theta) > 1:
                if cos_theta > 0:
                    cos_theta = 1
                else:
                    cos_theta = -1
            angle = np.arccos(cos_theta) * left_right

            self.offset.append(angle)
        return

    def prob_caculate(self, rank):
        prob = 1
        for i in range(self.vehicle_num):
            if i == rank:
                wpt = self.wpt[i]
                temp_target = wpt[3]  #
                yaw = self.UGV[i].theta
                car_dir = np.array([math.cos(yaw), math.sin(yaw)])
                s_dir = np.array([temp_target[0] - self.UGV[i].x, temp_target[1] - self.UGV[i].y])
                cos_theta = np.dot(car_dir, s_dir) / (np.linalg.norm(car_dir) * np.linalg.norm(s_dir))
                left_right = (abs(np.cross(car_dir, s_dir)) + 1e-10) / (np.cross(car_dir, s_dir) + 1e-10)
                if abs(cos_theta) > 1:
                    if cos_theta > 0:
                        cos_theta = 1
                    else:
                        cos_theta = -1
                angle = np.arccos(cos_theta) * left_right
                steer = self.steer[rank]
                prob = norm.pdf(steer,angle,1*0.25) / norm.pdf(angle,angle,1*0.25)

        return prob



