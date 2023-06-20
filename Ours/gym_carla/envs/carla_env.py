#!/usr/bin/env python

# Copyright (c) 2019: Jianyu Chen (jianyuchen@berkeley.edu)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division
import logging
import copy
import numpy as np
import pygame
import random
import time
from skimage.transform import resize
from ..util.misc import get_acceleration
import gym
from gym import spaces
from gym.utils import seeding
import carla
from utils.utils import AgentState
import math
from gym_carla.envs.render import BirdeyeRender
from agents.navigation.global_route_planner import GlobalRoutePlanner
from gym_carla.envs.misc import *
from utils.PID import IncreasPID as PID
import matplotlib.pyplot as plt
import cv2
import random


class CarlaEnv(gym.Env):
    """An OpenAI gym wrapper for CARLA simulator."""

    def __init__(self, params):
        # parameters
        self.display_size = params['display_size']  # rendering screen size
        self.max_past_step = params['max_past_step']
        self.number_of_vehicles = params['number_of_vehicles']
        self.dt = params['dt']
        self.max_time_episode = params['max_time_episode']
        self.max_waypt = params['max_waypt']
        self.obs_range = params['obs_range']
        self.d_behind = params['d_behind']
        self.obs_size = int(self.obs_range / 0.125)  # 256
        self.out_lane_thres = params['out_lane_thres']
        self.desired_speed = params['desired_speed']
        self.max_ego_spawn_times = params['max_ego_spawn_times']
        self.display_route = params['display_route']
        self.speed_last = None
        self.actorlist = []
        self.ego_state = AgentState()
        self.ap_vehicles = []
        self.long = params['long']
        self.destroy = params['temp_destroy']
        trans = params['temp_transform']
        self.temp_transform = []
        for i in range(len(trans)):
            temp_transform = carla.Transform(
                carla.Location(x=trans[i][0], y=trans[i][1], z=0.2),
                carla.Rotation(yaw=trans[i][2]))
            temp_transform.location.z = 10.2
            self.temp_transform.append(temp_transform)

        self.cnt = 0

        # Connect to carla server and get world object
        print('connecting to Carla server...')
        self.client = carla.Client('localhost', params['port'])
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(params['town'])
        print('Carla server connected!')
        self.map = self.world.get_map()

        # Set weather
        self.world.set_weather(carla.WeatherParameters.ClearNoon)

        # Get spawn points
        self.vehicle_spawn_points = list(self.world.get_map().get_spawn_points())

        # Create the ego vehicle blueprint
        self.ego_bp = self._create_vehicle_bluepprint(params['ego_vehicle_filter'], color='49,8,8')
        self.opponent_bp = self._create_vehicle_bluepprint(params['opponent_vehicle_filter'])

        # Collision sensor
        self.collision_hist = []  # The collision history
        self.collision_hist_l = 1  # collision history length
        self.collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')

        # Camera sensor
        self.camera_img = np.zeros((self.obs_size, self.obs_size, 3), dtype=np.uint8)
        self.camera_trans = carla.Transform(carla.Location(x=0.8, z=1.7))
        self.camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        # Modify the attributes of the blueprint to set image resolution and field of view.
        self.camera_bp.set_attribute('image_size_x', str(self.obs_size))
        self.camera_bp.set_attribute('image_size_y', str(self.obs_size))
        self.camera_bp.set_attribute('fov', '110')
        # Set the time in seconds between sensor captures
        self.camera_bp.set_attribute('sensor_tick', '0.05')

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0

        self.noise = params['noise']
        trans = params['ego_transform']
        self.ego_transform = []
        if self.noise:
            for i in range(len(trans)):
                ego_transform = carla.Transform(
                    carla.Location(x=trans[i][0], y=trans[i][1], z=0.2),
                    carla.Rotation(yaw=trans[i][2]))
                ego_transform = self.map.get_waypoint(ego_transform.location).transform
                noise = random.randrange(-5,5) * 1
                yaw = ego_transform.rotation.yaw / 180 * math.pi
                ego_transform.location.x += noise * math.cos(yaw)
                ego_transform.location.y += noise * math.sin(yaw)
                ego_transform.location.z = 4.2
                self.ego_transform.append(ego_transform)
        else:
            for i in range(len(trans)):
                ego_transform = carla.Transform(
                    carla.Location(x=trans[i][0], y=trans[i][1], z=0.2),
                    carla.Rotation(yaw=trans[i][2]))
                ego_transform = self.map.get_waypoint(ego_transform.location).transform
                ego_transform.location.z = 4.2
                self.ego_transform.append(ego_transform)

        trans = params['target_transform']
        self.target_transform = []
        for i in range(len(trans)):
            target_transform = carla.Transform(
                carla.Location(x=trans[i][0], y=trans[i][1], z=0.2),
                carla.Rotation(yaw=trans[i][2]))
            self.target_transform.append(target_transform)

        trans = params['temp_target']
        self.temp_target = []
        for i in range(len(trans)):
            target_transform = carla.Transform(
                carla.Location(x=trans[i][0], y=trans[i][1], z=0.2),
                carla.Rotation(yaw=trans[i][2]))
            self.temp_target.append(target_transform)
        self.temp_target.append(self.target_transform[1])
        # Initialize the renderer
        self._init_renderer()

    def reset(self):
        # Clear sensor objects
        self.collision_sensor = None
        self.camera_sensor = None

        # Delete sensors, vehicles and walkers
        self._clear_all_actors()
        self.actorlist = []
        # Disable sync mode
        self._set_synchronous_mode(False)

        # Spawn surrounding vehicles
        # random.shuffle(self.vehicle_spawn_points)
        count = self.number_of_vehicles
        if count > 0:
            for spawn_point in self.vehicle_spawn_points:
                if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
                    count -= 1
                if count <= 0:
                    break
        while count > 0:
            if self._try_spawn_random_vehicle_at(random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
                count -= 1
        # Get actors polygon list
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)

        self.egos = []
        # Spawn the ego vehicles
        for i in range(len(self.ego_transform)):
            ego_spawn_times = 0
            while True:
                if ego_spawn_times > self.max_ego_spawn_times:
                    self.reset()
                if i == 0:
                    if self._try_spawn_ego_vehicle_at(self.ego_transform[i]):
                        break
                    else:
                        ego_spawn_times += 1
                        time.sleep(0.1)
                else:
                    if self._try_spawn_other_vehicle_at(self.ego_transform[i]):
                        break
                    else:
                        ego_spawn_times += 1
                        time.sleep(0.1)
        #self.egos.append(self.ego)
        # Add collision sensor
        self.collision_sensor = self.world.spawn_actor(self.collision_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(lambda event: get_collision_hist(event))
        self.actorlist.append(self.collision_sensor)

        def get_collision_hist(event):
            impulse = event.normal_impulse
            intensity = np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
            self.collision_hist.append(intensity)
            if len(self.collision_hist) > self.collision_hist_l:
                self.collision_hist.pop(0)

        self.collision_hist = []

        # Add camera sensor
        self.camera_sensor = self.world.spawn_actor(self.camera_bp, self.camera_trans, attach_to=self.ego)
        self.camera_sensor.listen(lambda data: get_camera_img(data))
        self.actorlist.append(self.camera_sensor)

        def get_camera_img(data):
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.camera_img = array

        # Update timesteps
        self.time_step = 0
        self.reset_step += 1
        self.PID1 = PID()
        self.PID2 = PID()
        self.PID = [self.PID1,self.PID2]

        # Enable sync mode
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)

        self.waypoints_all = []
        for i in range(len(self.egos)):
            self.waypoints = []
            self.routeplanner = GlobalRoutePlanner(self.map, 2)
            self.temp = self.routeplanner.trace_route(self.egos[i].get_transform().location,
                                                          self.target_transform[i].location)
            for _, (waypoint, _) in enumerate(self.temp):
                self.waypoints.append(
                    [waypoint.transform.location.x, waypoint.transform.location.y, waypoint.transform.rotation.yaw])
            self.waypoints_all.append(self.waypoints)

        # Set ego information for render
        self.birdeye_render.set_hero(self.ego, self.ego.id)

        self.cmd = list(np.zeros(len(self.egos)))

        return self._get_obs()

    def step(self, action):
        self.v_now = []
        acts = []
        for i in range(len(self.egos)):
            v_now = self.egos[i].get_velocity()
            v_now = math.sqrt(v_now.x ** 2 + v_now.y ** 2)
            self.v_now.append(v_now)

            v_cmd = action[i][0]
            steer = action[i][1]
            delta = self.PID[i].run(v_cmd, v_now)
            self.cmd[i] += delta
            # print(f"current speed:{v_now}, delta:{delta},cmd:{self.cmd},step:{self.time_step}")

            if self.cmd[i] > 0:
              throttle = self.cmd[i]
              if throttle > 1.0:
                throttle = 1.0
              brake = 0
            else:
              throttle = 0
              brake = -self.cmd[i]
              if brake > 1.0:
                brake = 1.0
            # Apply control
            # print(f"acc:{float(throttle)}    steer:{float(steer)}")
            act = carla.VehicleControl(throttle=float(throttle), steer=float(steer), brake=float(brake))
            acts.append(act)

            # Append actors polygon list
            #vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
           # self.vehicle_polygons.append(vehicle_poly_dict)
            #while len(self.vehicle_polygons) > self.max_past_step:
             #   self.vehicle_polygons.pop(0)
        for i in range(len(self.egos)):
            self.egos[i].apply_control(acts[i])
        #self.ego.apply_control(acts[0])
        self.world.tick()
        self.check_spaw()
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        while len(self.vehicle_polygons) > self.max_past_step:
           self.vehicle_polygons.pop(0)
        self.waypoints_all = []

        for i in range(len(self.egos)):
            self.waypoints = []
            #self.routeplanner = GlobalRoutePlanner(self.map, 2)
            self.temp = self.routeplanner.trace_route(self.egos[i].get_transform().location, self.target_transform[i].location)
            for _, (waypoint, _) in enumerate(self.temp):
                self.waypoints.append(
                    [waypoint.transform.location.x, waypoint.transform.location.y, waypoint.transform.rotation.yaw])
            self.waypoints_all.append(self.waypoints)
            """
            x = self.waypoints_all[i][0][0]
            y = self.waypoints_all[i][0][1]
            if math.sqrt(x**2+y**2) < 0.25:
                self.waypoints_all[i].pop(0)
            self.waypoints = self.waypoints_all[0]
            """


        # obs information
        info = {
            'waypoints': self.waypoints_all
        }

        # Update timesteps
        self.time_step += 1
        self.total_step += 1

        spectator = self.world.get_spectator()
        transform = self.ego.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50),
                                                carla.Rotation(pitch=-90)))
        #for i in range(len(self.egos)):
        #    self.draw_waypoints(self.waypoints_all[i])

        #for i in range(len(self.egos)):
       #     self.draw_arrow(self.egos[i].get_transform().location,self.waypoints_all[i][4])

        return (self._get_obs(), 0, self._terminal(), 0)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode):
        pass

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        """Create the blueprint for a specific actor type.
    Args:
      actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.
    Returns:
      bp: the blueprint object of carla.
    """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if
                                                     int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def _init_renderer(self):
        """Initialize the birdeye view renderer.
    """
        pygame.init()
        self.display = pygame.display.set_mode(
            (self.display_size * 2, self.display_size),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        pixels_per_meter = self.display_size / self.obs_range
        pixels_ahead_vehicle = (self.obs_range / 2 - self.d_behind) * pixels_per_meter
        birdeye_params = {
            'screen_size': [self.display_size, self.display_size],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle
        }
        self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

    def _set_synchronous_mode(self, synchronous=True):
        """Set whether to use the synchronous mode.
    """
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """Try to spawn a surrounding vehicle at specific transform with random bluprint.
    Args:
      transform: the carla transform object.
    Returns:
      Bool indicating whether the spawn is successful.
    """
        blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        self.barrier = vehicle
        if vehicle is not None:
            vehicle.set_autopilot()
            return True
        self.ap_vehicles.append(vehicle)
        return False

    def _try_spawn_other_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.
    Args:
      transform: the carla transform object.
    Returns:
      Bool indicating whether the spawn is successful.
    """
        vehicle = None
        # Check if ego position overlaps with surrounding vehicles
        overlap = False
        for idx, poly in self.vehicle_polygons[-1].items():
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
            if dis > 8:
                continue
            else:
                overlap = True
                break
        #blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=[4])
        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)

        if vehicle is not None:
            self.egos.append(vehicle)
            self.actorlist.append(vehicle)
            return True

        return False

    def _try_spawn_ego_vehicle_at(self, transform):
        """Try to spawn the ego vehicle at specific transform.
    Args:
      transform: the carla transform object.
    Returns:
      Bool indicating whether the spawn is successful.
    """
        vehicle = None
        # Check if ego position overlaps with surrounding vehicles
        overlap = False
        for idx, poly in self.vehicle_polygons[-1].items():
            poly_center = np.mean(poly, axis=0)
            ego_center = np.array([transform.location.x, transform.location.y])
            dis = np.linalg.norm(poly_center - ego_center)
            if dis > 8:
                continue
            else:
                overlap = True
                break

        if not overlap:
            vehicle = self.world.try_spawn_actor(self.ego_bp, transform)
        #  print(vehicle)

        if vehicle is not None:
            self.ego = vehicle
            self.egos.append(self.ego)
            self.actorlist.append(self.ego)
            return True

        return False

    def _get_actor_polygons(self, filt):
        """Get the bounding box polygon of actors.
    Args:
      filt: the filter indicating what type of actors we'll look at.
    Returns:
      actor_poly_dict: a dictionary containing the bounding boxes of specific actors.
    """
        actor_poly_dict = {}
        for actor in self.world.get_actors().filter(filt):
            # Get x, y and yaw of the actor
            trans = actor.get_transform()
            x = trans.location.x
            y = trans.location.y
            yaw = trans.rotation.yaw / 180 * np.pi
            # Get length and width
            bb = actor.bounding_box
            l = bb.extent.x
            w = bb.extent.y
            # Get bounding box polygon in the actor's local coordinate
            poly_local = np.array([[l, w], [l, -w], [-l, -w], [-l, w]]).transpose()
            # Get rotation matrix to transform to global coordinate
            R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
            # Get global bounding box polygon
            poly = np.matmul(R, poly_local).transpose() + np.repeat([[x, y]], 4, axis=0)
            actor_poly_dict[actor.id] = poly
        return actor_poly_dict

    def _get_obs(self):
        """Get the observations."""
        ## Birdeye rendering
        self.birdeye_render.vehicle_polygons = self.vehicle_polygons
        self.birdeye_render.waypoints = self.waypoints

        # birdeye view with roadmap and actors
        birdeye_render_types = ['roadmap', 'actors']
        if self.display_route:
            birdeye_render_types.append('waypoints')
        self.birdeye_render.render(self.display, birdeye_render_types)
        birdeye = pygame.surfarray.array3d(self.display)
        birdeye = birdeye[0:self.display_size, :, :]
        birdeye = display_to_rgb(birdeye, self.obs_size)
        #img = cv2.cvtColor(birdeye, cv2.COLOR_BGR2RGB)
        #cv2.imwrite("../data/" + f'{0}.png', birdeye)

        # Display birdeye image
        birdeye_surface = rgb_to_display_surface(birdeye, self.display_size)
        self.display.blit(birdeye_surface, (0, 0))

        ## Display camera image
        camera = resize(self.camera_img, (self.obs_size, self.obs_size)) * 255
        camera_surface = rgb_to_display_surface(camera, self.display_size)
        self.display.blit(camera_surface, (self.display_size * 1, 0))

        # Display on pygame
        pygame.display.flip()

        # State observation
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw / 180 * np.pi
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x ** 2 + v.y ** 2)

        self.ego_state.front_view = camera.astype(np.uint8)
        self.ego_state.pos = [ego_x, ego_y, ego_yaw]
        self.ego_state.speed = speed

        return self.ego_state

    def _get_reward(self):
        """Calculate the step reward."""
        # reward for speed tracking
        v = self.ego.get_velocity()
        speed = np.sqrt(v.x ** 2 + v.y ** 2)
        r_speed = -abs(speed - self.desired_speed)

        # reward for collision
        r_collision = 0
        if len(self.collision_hist) > 0:
            r_collision = -1

        # reward for steering:
        r_steer = -self.ego.get_control().steer ** 2

        # reward for out of lane
        ego_x, ego_y = get_pos(self.ego)
        dis, w = get_lane_dis(self.waypoints, ego_x, ego_y)
        r_out = 0
        if abs(dis) > self.out_lane_thres:
            r_out = -0.5

        # longitudinal speed
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)

        # cost for too fast
        r_fast = 0
        if lspeed_lon > self.desired_speed:
            r_fast = -1

        # cost for lateral acceleration
        r_lat = - abs(self.ego.get_control().steer) * lspeed_lon ** 2

        r = 200 * r_collision + 100 * lspeed_lon + 10 * r_fast + 1 * r_out + r_steer * 500 + 0.2 * r_lat - 0.1

        return r

    def _terminal(self):
        """Calculate whether to terminate the current episode."""
        # Get ego obs
        if self.time_step > 1000:
            pass
        ego_x, ego_y = get_pos(self.ego)

        # If collides
        if len(self.collision_hist) > 0:
            print("crash happened!")
            return True

        # If out of lane
        #, _ = get_lane_dis(self.waypoints, ego_x, ego_y)
        #if abs(dis) > self.out_lane_thres:
        #    return True
        # print(f"dis:{dis}")

        return False

    def _clear_all_actors(self):
        """Clear specific actors."""
        """
    for actor_filter in actor_filters:
      for actor in self.world.get_actors().filter(actor_filter):
        if actor.is_alive:
          if actor.type_id == 'controller.ai.walker':
            actor.stop()
          actor.destroy()
    """
        for actor in self.actorlist:
            actor.destroy()
        # print(self.actorlist)
        # print(self.world.get_actors().filter('sensor.camera.rgb'))

    def draw_waypoints(self, waypoints):
        for waypoint in waypoints:
            self.world.debug.draw_point(carla.Location(x=waypoint[0],y=waypoint[1],z=1),
                                    color=carla.Color(r=255, g=0, b=0),
                                    life_time=0.1)

    def draw_arrow(self,s,e):
       # print(s,e)
        start = s
        end = carla.Location(x=e[0],y=e[1],z=1)
        self.world.debug.draw_arrow(start, end, life_time=0.1)

    def check_spaw(self):
        if self.long:

            trans = self.ego.get_transform()
            x = trans.location.x
            y = trans.location.y

            if self.cnt < 3:
                print(x, self.destroy[self.cnt])
                if x > self.destroy[self.cnt]:
                    self.egos[0].destroy()
                    self.egos = []
                    self._try_spawn_other_vehicle_at(self.temp_transform[self.cnt])
                    self.target_transform[0] = self.temp_target[self.cnt]
                    self.cnt += 1
                    self.egos.append(self.ego)
                    print("change!")
            else:
                print(y, self.destroy[self.cnt])
                if y > self.destroy[self.cnt]:
                    self.egos[0].destroy()
                    self.egos = []
                    self._try_spawn_other_vehicle_at(self.temp_transform[self.cnt])
                    self.target_transform[0] = self.temp_target[self.cnt]
                    self.cnt += 1
                    self.egos.append(self.ego)
                    print(" final change!")

            pass

