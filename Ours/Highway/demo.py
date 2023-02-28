import highway_env
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.envs.common.action import Action
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


LANES = 2
ANGLE = 0
START = 0
LENGHT = 200
SPEED_LIMIT = 30
SPEED_REWARD_RANGE = [10, 30]
COL_REWARD = -1
HIGH_SPEED_REWARD = 0
RIGHT_LANE_REWARD = 0
DURATION = 100.0


class myEnv(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                'observation': {
                    'type': 'Kinematics',
                    "absolute": False,
                    "normalize": False
                },
                'action': {'type': 'DiscreteMetaAction'},

                "reward_speed_range": SPEED_REWARD_RANGE,
                "simulation_frequency": 20,
                "policy_frequency": 20,
                "centering_position": [0.3, 0.5],
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()


    def _create_road(self) -> None:
        self.road = Road(
            network=RoadNetwork.straight_road_network(LANES, speed_limit=SPEED_LIMIT),
            np_random=self.np_random,
            record_history=False,
        )

	# 创建车辆
    def _create_vehicles(self) -> None:

        vehicle = Vehicle.create_random(self.road, speed=23, lane_id=1, spacing=0.3)
        vehicle = self.action_type.vehicle_class(
            self.road,
            vehicle.position,
            vehicle.heading,
            vehicle.speed,
        )
        self.vehicle = vehicle
        self.road.vehicles.append(vehicle)

        vehicle = Vehicle.create_random(self.road, speed=30, lane_id=1, spacing=0.35)
        vehicle = self.action_type.vehicle_class(
            self.road,
            vehicle.position,
            vehicle.heading,
            vehicle.speed,
        )
        self.road.vehicles.append(vehicle)

	# 重写的奖励函数，仅考虑车辆碰撞影响
    def _reward(self, action: Action) -> float:
        reward = 0

        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )

        if self.vehicle.crashed:
            reward = -1
        elif lane == 0:
            reward += 1

        reward = 0 if not self.vehicle.on_road else reward

        return reward

    def _is_terminal(self) -> bool:
        return (
            self.vehicle.crashed
            or self.time >= DURATION
            or (False and not self.vehicle.on_road)
        )


if __name__ == '__main__':
    env = myEnv()
    obs = env.reset()

    eposides = 10
    rewards = 0
    for eq in range(eposides):
        obs = env.reset()
        env.render()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            env.render()
            rewards += reward
            print(obs)
        print(rewards)

