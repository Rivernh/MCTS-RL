import sys
import gym
import highway_env
sys.path.insert(0,r'/home/ylh/MCTS-RL/Ours')
from Module.mcts_pure import MCTSPlayer as MCTS_Pure

# Create environment
env = gym.make("highway-v0")

env.configure({
    'observation': {
        'type': 'Kinematics',
        "absolute": False,
        "normalize": False
    },
    #'action': {'type': 'DiscreteMetaAction'},
    'action': {'type': 'ContinuousAction'},
    'simulation_frequency': 15,
    'policy_frequency': 10,
    'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
    'screen_width': 600,
    'screen_height': 150,
    'centering_position': [0.3, 0.5],
    'scaling': 5.5,
    'show_trajectories': False,
    'render_agent': True,
    'offscreen_rendering': False,
    'manual_control': False,
    'real_time_rendering': False,
    'lanes_count': 5,  # 车道数
    'vehicles_count': 10,  # 周围车辆数
    'controlled_vehicles': 10,
    'initial_lane_id': None,
    'duration': 100,
    'ego_spacing': 2,
    'vehicles_density': 1,
    'collision_reward': -1,
    'right_lane_reward': 0.1,
    'high_speed_reward': 0.4,
    'lane_change_reward': 0,
    'reward_speed_range': [20, 30],
    'offroad_terminal': False,
    'lane_from': 1
})

eposides = 10
rewards = 0
mcts_player = MCTS_Pure(c_puct=1, n_playout=5000)
for eq in range(eposides):
    obs = env.reset()
    print(obs)
    env.render()
    done = False
    while not done:
        action = mcts_player.get_action(obs)
        print(f"action:{action}")
        obs, reward, done, info = env.step(action)
        # env.close()  # 关闭环境
        env.render()
        rewards += reward
    print(rewards)

