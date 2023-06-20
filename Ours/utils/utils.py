import numpy as np
import sys 
"""
sys.path.append(r'/home/ylh/MCTS-Carla/scripts/CARLA')
sys.path.insert(0,r'/home/ylh/MCTS-Carla/scripts/Ours')
"""
from agents.navigation.basic_agent import BasicAgent
print(sys.path)
#from Module.mcts_pure import MCTSPlayer
import carla
import math
from utils.PID import IncreasPID
from collections import deque
from queue import Queue
#available action of accelerate
action_num = 10
action_available = np.arange(-5,5,10 / action_num)
action_index = np.arange(0, action_num)

#智能体记录的状态
class AgentState:
    def __init__(self):
        self.front_view = np.zeros((256, 256, 3), dtype=np.uint8)
        self.speed = carla.Vector3D()
        self.acc = carla.Vector3D()
        self.pos = carla.Transform()

#some para of the exp1
class Mypara:
    """
    some paras used in the paper
    """
    def __init__(self):
        self.n_agent = 2
        #accelerate speed
        self.N = 5
        self.cput = 1.0
        self.rou = 1.0
        self.sigma = 0.4
        self.theta = np.array([1, 1])
        self.alpha = 0.1
        self.beta = 0.1
        self.amax = 5
        self.vmx = 20
        self.action_available = np.arange(-5,5,1)
        self.n_action = 10
        self.courtesy_weight = 1 / (self.n_agent-1)

#description of the agents
class Assistagent(BasicAgent):
    """"
    description of the agents
    """
    def __init__(self, vehicle):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param ignore_traffic_light: boolean to ignore any traffic light
            :param behavior: type of agent to apply
        """
        super(Assistagent, self).__init__(vehicle, target_speed = 80, opt_dict = {'ignore_stop_signs':True, 'ignore_traffic_lights':True})
        self.para = Mypara()
        self.state_Nstep = Queue(maxsize = self.para.N)#AgentState()
        self.ego_reward = np.zeros((self.para.n_action))
        self.cour_reward = np.zeros((self.para.n_action))
        self.driving_style = np.ones((self.para.n_action)) * 0.5
        self.total_reward = np.zeros((self.para.n_action))

    def ego_reward_update(self, order):
        a = self.agent_state[3 * order, :]
        v = self.agent_state[3 * order + 1, :]
        reward = self.para.theta @ np.array([sum(np.exp(-self.para.alpha * a * a)),sum(1 - np.exp(-self.para.beta * v * v))])
        self.ego_reward[order] = reward
        return self.ego_reward[order]

    def cou_reward_update(self, order):
        self.cour_reward = (sum(self.ego_reward) - self.ego_reward[order]) * self.para.courtesy_weight
        return self.cour_reward

    def set_target_transform(self, target_transform):
        self.target_transform = target_transform
        return True

    def run(self):
        destination = self.target_transform.location
        self.set_destination(destination)
        control = carla.VehicleControl(brake = 1.0)
        if self.done():
            print("The target has been reached, stopping the simulation")
        else:
            control=self.run_step()
            self._vehicle.apply_control(control)
            spectator = self._world.get_spectator()
            transform = self._vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + 
            carla.Location(-15 * math.cos((transform.rotation.yaw)/180*math.pi),-20 * math.sin((transform.rotation.yaw)/180*math.pi),10),
            carla.Rotation(-20,transform.rotation.yaw,transform.rotation.roll)))
        return control



    def safety_check(action, img, state):
        """
        input para:
        action:all candidate action list
        img:current a w*h img
        obs:current obs vector including ego agent and another agent also drived by my algorithm
        output para:legal action list
        """
        legal_actions = action_available
        return legal_actions

#description of the agents with mcts
class MCTSagent(BasicAgent):
    """"
    description of the agents with mcts
    """
    def __init__(self, vehicle, env, behavior = 0.5):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param ignore_traffic_light: boolean to ignore any traffic light
            :param behavior: type of agent to apply
        """

        super(MCTSagent, self).__init__(vehicle)
        self.para = Mypara()
        self.behavior = behavior
        #self.MCTSPlayer = MCTSPlayer(env, c_puct=5, n_playout=10000)
        self.state_input = np.zeros((self.para.vector_length, self.para.N))
        self.real_state = np.zeros((4,3))#智能体的坐标 姿态 速度 加速度3+3+3+3
        self.ego_reward = np.zeros((self.para.n_action))
        self.cour_reward = np.zeros((self.para.n_action))
        self.driving_style = np.ones((self.para.n_action)) * 0.5
        self.total_reward = np.zeros((self.para.n_action))

    def ego_reward_update(self, order):
        a = self.agent_state[3 * order, :]
        v = self.agent_state[3 * order + 1, :]
        reward = self.para.theta @ np.array([sum(np.exp(-self.para.alpha * a * a)),sum(1 - np.exp(-self.para.beta * v * v))])
        self.ego_reward[order] = reward
        return self.ego_reward[order]

    def cou_reward_update(self, order):
        self.cour_reward = (sum(self.ego_reward) - self.ego_reward[order]) * self.para.courtesy_weight
        return self.cour_reward

    def set_target_transform(self, target_transform):
        self.target_transform = target_transform
        return True

    def run(self):
        destination = self.target_transform.location
        self.set_destination(destination)
        control = None
        PIDer = PID()
        if self.done():
            print("The target has been reached, stopping the simulation")
        else:
            move = self.MCTSPlayer.get_action()
            control = carla.VehicleControl()
            control.throttle = PIDer.run(move, self._vehicle.get_velocity())
            self._vehicle.apply_control(control)
            spectator = self._world.get_spectator()
            transform = self._vehicle.get_transform()
            spectator.set_transform(carla.Transform(transform.location + 
            carla.Location(-15 * math.cos((transform.rotation.yaw)/180*math.pi),-20 * math.sin((transform.rotation.yaw)/180*math.pi),10),
            carla.Rotation(-20,transform.rotation.yaw,transform.rotation.roll)))
        return control

    def done(self):
        """Check whether the agent has reached its destination."""
        target_dis = self.target_transform.location.distance(self._vehicle.get_location())
        if target_dis<5:
            self._vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0, brake=1, hand_brake=False, reverse=False))
            print('arrive target location!')
            return True
        return False

    def safety_check(action, img, state):
        """
        input para:
        action:all candidate action list
        img:current a w*h img
        obs:current obs vector including ego agent and another agent also drived by my algorithm
        output para:legal action list
        """
        legal_actions = action_available
        return legal_actions



if __name__ == '__main__':
    """"
        import torch
    print('Torch Version:',torch.__version__)
    print('CUDA GPU check:',torch.cuda.is_available())
    if(torch.cuda.is_available()):
        print('CUDA GPU num:', torch.cuda.device_count())
        n=torch.cuda.device_count()
    while n > 0:
        print('CUDA GPU name:', torch.cuda.get_device_name(n-1))
        n -= 1
    print('CUDA GPU index:', torch.cuda.current_device())
    """
    #a = carla.Vector3D(1,2,3)
    #b = np.array([a.x,a.y,a.z])
    #print(b)
    a = Queue(maxsize=5)
    x = AgentState()
    a.put([2])
    a.put(x)
    
    print(a.get())
    print(a.get())