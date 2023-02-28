import numpy as np
import math
import matplotlib.pyplot as plt
import time
import itertools

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
    info = (x, y, yaw, wheel_base, l, w)
    return info

def overlap(box1,box2):#TODO 加入已知矩形四个顶点，检测是否碰撞
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
    crash = True
    #box1
    for i in range(4):
        a = box1[i]
        b = box1[(i + 1) % 4]
        cnt = 0
        for point in box2:
            cnt += 1
            if my_caculate(a,b,point):
                break
            if cnt == 4:
                crash = False
                return crash
    #box2
    for i in range(4):
        a = box2[i]
        b = box2[(i + 1) % 4]
        cnt = 0
        for point in box1:
            cnt += 1
            if my_caculate(a,b,point):
                break
            if cnt == 4:
                crash = False
                return crash
    return crash

def inside_range():#TODO 加入在规划轨迹一定范围内的的检测，便于后续加入steer维度的动作

    return False

class UGV_model:
    def __init__(self, info, T):  # L:wheel base
        self.x = info[0] # X
        self.y = info[1]  # Y
        self.theta = info[2]  # headding
        self.wb = info[3]  # wheel base
        self.l = info[4]
        self.w = info[5]

        self.dt = T  # decision time periodic

    def update(self, vt, deltat):  # update ugv's state
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
    def __init__(self, vehicle,dt=0.1):  # L:wheel base
        self.vehicle = vehicle
        info = my_get_info(self.vehicle)
        self.x = info[0]  # X
        self.y = info[1]  # Y
        self.theta = info[2]  # headding
        self.wb = info[3]  # wheel base
        self.l = info[4]
        self.w = info[5]
        self.dt = dt

        v = self.vehicle.get_velocity()
        self.v = math.sqrt(v.x ** 2 + v.y ** 2)

    def update(self):  # update ugv's state
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
    def __init__(self, vehicle, wpt, ap_vehicle = None, dt = 0.1):
        #self.availables = list(itertools.product(np.array([0,3,6,9,12]), np.array([0,-0.2,-0.4,0.2,0.4])))
        self.availables = np.array([0,2.5,5,7.5,10])
        self.dt = dt
        self.vehicle = vehicle
        self.ap_vehicle = ap_vehicle
        self.vehicle_num = len(self.vehicle)
        self.AP_num = len(self.ap_vehicle)
        self.total = self.vehicle_num + self.AP_num
        self.step = None
        self.crash = np.zeros(self.vehicle_num)
        self.wpt = wpt
        #print(f"vehicle_num:{self.vehicle_num}")

    def reset(self):
        self.step = 0
        self.end = False
        self.UGV = []
        self.AP = []
        self.speed = []
        self.acc = [0]
        while len(self.UGV) < self.vehicle_num:
            v = self.vehicle[len(self.UGV)].get_velocity()
            v = math.sqrt(v.x ** 2 + v.y ** 2)
            self.speed.append(v)
            info = my_get_info(self.vehicle[len(self.UGV)])
            self.UGV.append(UGV_model(info, self.dt))
            #if len(self.speed):
            #    print("successfully add speed!")

        while len(self.AP) < self.AP_num:
            self.AP.append(ap_model(self.ap_vehicle[len(self.AP)]))
            #if len(self.speed):
            #    print("successfully add speed!")

    # do a move according to the action for one car
    def do_move(self, action):
        #v = action[0]
        #steer = action[1]

        v = action

        temp_target = self.wpt[1]  #
        yaw = self.UGV[0].theta
        car_dir = np.array([math.cos(yaw), math.sin(yaw)])
        s_dir = np.array([temp_target[0] - self.UGV[0].x, temp_target[1] - self.UGV[0].y])
        cos_theta = np.dot(car_dir, s_dir) / (np.linalg.norm(car_dir) * np.linalg.norm(s_dir))
        left_right = abs(np.cross(car_dir, s_dir)) / np.cross(car_dir, s_dir)
        self.angle = np.arccos(cos_theta) * left_right

        steer = self.angle * 1.5
        if steer > 0.9:
            steer = 0.9
        elif steer < -0.9:
            steer = -0.9


        self.acc[0] = abs(v-self.speed[0])
        self.UGV[0].update(v, steer)
        self.speed[0] = v
        #print("move done!")
        return

    # run an episode until end keeping the speed after the action for one agent
    def run_episode(self):
        while not self.end:
            self.UGV[0].update(self.speed[0], 0)
            self.AP[0].update()
            self.end = self.terminal()
            self.step += 1
        return self.get_reward()

    # get the reward of the action node
    def get_reward(self):
        reward = []
        for i in range(self.vehicle_num):
            speed_reward = 1 - np.exp(-0.025*(self.speed[i] ** 2))
            acc_reward = np.exp(-0.01*self.dt*(self.acc[i] ** 2))
            acc_reward = self.acc[i] / 10
            task_reward = self.task()
            rew = speed_reward + task_reward
            reward.append(rew)

        return reward

    # check if the episode is end and update the UGV state flag in the env
    def terminal(self):
        self.crash_check()
        if self.step >= 5:
            return True
        if self.crash:
            return True
        return False

    def task(self):
        rew = 0
        if self.crash:
            rew = -0.5 * (5 - self.step)
        return rew

    def crash_check(self):
        crash = False
        for i in range(self.vehicle_num):
            for j in range(self.total-1-i):
                if i+j+1 >= self.vehicle_num:
                    crash = overlap(self.UGV[i].box,self.AP[i+j+1-self.vehicle_num].box)
                #与周围车辆非控制的检测
                    #print(f"dis:{dis} thre:{thre}")
                else:
                    crash = overlap(self.UGV[i].box, self.UGV[i+j+1].box)
                    crash = False #与其他UGV检测
                self.crash = crash
                if crash:
                    print("carshed!")
                    break

    def off_track(self):
        """
        计算每个智能体偏离预定轨迹的距离，来控制循迹
        """

        dis = []
        wpt = []
        a = np.array(wpt[1])
        b = np.array(wpt[2])
        p = np.array([self.UGV[0].x, self.UGV[0].y])
        s = b-a
        c = p-a
        cos_theta = np.dot(c, s) / (np.linalg.norm(c) * np.linalg.norm(s))
        d = np.linalg.norm(c) * math.sqrt(1-cos_theta**2)
        dis.append(d)
        return dis


    """检查距离判断碰撞
    def crash_check(self):
        crash = False
        for i in range(self.vehicle_num):
            for j in range(self.total-1-i):
                if i+j+1 >= self.vehicle_num:
                    dis = (self.UGV[i].x-self.AP[i+j+1-self.vehicle_num].x)**2 + (self.UGV[i].y-self.AP[i+j+1-self.vehicle_num].y)**2
                    thre = (self.UGV[i].w + self.AP[i+j+1-self.vehicle_num].w)**2 + (self.UGV[i].l + self.AP[i+j+1-self.vehicle_num].l)**2
                    if dis <= thre:
                        crash = True
                #与周围车辆非控制的检测
                    #print(f"dis:{dis} thre:{thre}")
                else:
                    crash = False #与其他UGV检测
                self.crash = crash
                if crash:
                    print("carshed!")
                    break
    """






























