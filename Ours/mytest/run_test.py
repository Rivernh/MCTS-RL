
import glob
import os
import sys
from types import LambdaType
from collections import deque
from collections import namedtuple
import math
 
import carla
import random 
import time
import numpy as np
sys.path.append(r'/home/ylh/MCTS-Carla/scripts/CARLA')
sys.path.append(r'/home/ylh/MCTS-Carla/scripts/Ours')
from agents.navigation.basic_agent import BasicAgent
from utils.process import start_process
from agents.navigation.controller import PIDLongitudinalController
from agents.navigation.global_route_planner import GlobalRoutePlanner
from utils.utils import Assistagent

start_process()
client = carla.Client('localhost',2000)
client.set_timeout(10.0)

world = client.load_world('town05')
blueprint_library = world.get_blueprint_library()
model_3 = blueprint_library.filter('model3')[0]
 
actor_list = []
transform = world.get_map().get_spawn_points()[90] #spwan_points共265个点，选第一个点作为初始化小车的位置
#random.choice(self.world.get_map().get_spawn_points())
vehicle = world.spawn_actor(model_3 , transform)
agent = Assistagent(vehicle)
#agent = BasicAgent(vehicle)
actor_list.append(vehicle)   
 
target_transform = world.get_map().get_spawn_points()[110]
print(target_transform.location.x)
print(target_transform.location.y)

#destination = [target_transform.location.x,target_transform.location.y,target_transform.location.z]
destination = target_transform.location
agent.set_destination(destination)

agent.set_target_transform(target_transform)
while True:
    if agent.done():
        print("Tagent.run(target_transform)he target has been reached, stopping the simulation")
        break
    agent.run()
 
for agent in actor_list:    
    agent.destroy()
print('done')
"""

while True:
    if agent.done():
        print("The target has been reached, stopping the simulation")
        break
    control=agent.run_step()
    vehicle.apply_control(control)
    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    spectator.set_transform(carla.Transform(transform.location + 
    carla.Location(-15 * math.cos((transform.rotation.yaw)/180*math.pi),-20 * math.sin((transform.rotation.yaw)/180*math.pi),10),
    carla.Rotation(-20,transform.rotation.yaw,transform.rotation.roll)))
 
for agent in actor_list:    
    agent.destroy()
print('done')

"""
"""
env_map = world.get_map()
grp = GlobalRoutePlanner(env_map,3)#3表示每隔3m采样一个路径点，路径点越密集，路径的精度越高

PID = PIDLongitudinalController(vehicle)

while True:
    route_way_point =grp.trace_route(vehicle.get_transform().location,target_transform.location)
 
    if len(route_way_point)>3:
        temp_target = route_way_point[1][0]#
        target_mat=temp_target.transform.get_matrix()
    else:
        temp_target = target_transform
        target_mat=temp_target.get_matrix()
    target_dis = target_transform.location.distance(vehicle.get_location())
    car_mat=vehicle.get_transform().get_matrix()    
    car_dir=np.array([car_mat[0][0],car_mat[1][0]])
    s_dir = np.array([target_mat[0][3]-car_mat[0][3],target_mat[1][3]-car_mat[1][3]])
 
    cos_theta=np.dot(car_dir,s_dir)/(np.linalg.norm(car_dir)*np.linalg.norm(s_dir))
 
    left_right = abs(np.cross(car_dir,s_dir))/np.cross(car_dir,s_dir)
    angle = np.arccos(cos_theta)*left_right
    vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=angle*1.5, brake=0.0, hand_brake=False, reverse=False))
   #time.sleep(0.2)
    v = vehicle.get_velocity()
    kmh = int(3.6 * np.sqrt(v.x**2 + v.y**2 + v.z**2))
    print('v:',kmh,'kmh\t','left_distance:',int(target_dis),'m')
    if target_dis<5:
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0, brake=1, hand_brake=False, reverse=False))
        print('arrive target location!')
        break
    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    spectator.set_transform(carla.Transform(transform.location + 
    carla.Location(-15 * math.cos((transform.rotation.yaw)/180*math.pi),-20 * math.sin((transform.rotation.yaw)/180*math.pi),10),
    carla.Rotation(-20,transform.rotation.yaw,transform.rotation.roll)))
 
for agent in actor_list:
    agent.destroy()
print('done')

"""
