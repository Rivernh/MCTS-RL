#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
"""
import glob
import os
import sys
from os.path import join, dirname
sys.path.insert(0, join(dirname(__file__), '../../'))

import simulator
simulator.load('/home/ylh/CARLA/CARLA_0.9.13')

sys.path.append('/home/ylh/CARLA/CARLA_0.9.13/PythonAPI/carla')

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

try:
    import pygame
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

from simulator import config, set_weather, add_vehicle
from agents.navigation.basic_agent import BasicAgent
from simulator.sensor_manager import SensorManager
from utils_.navigator_sim import get_map, get_nav, replan, close2dest
from utils_ import add_alpha_channel

def main():   
    world = None
    original_settings = None
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--generation',
        metavar='G',
        default='2',
        help='restrict to certain actor generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--rolename',
        metavar='NAME',
        default='hero',
        help='actor role name (default: "hero")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)
    
    try:

        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)

        world = client.load_world('Town01')
        weather = carla.WeatherParameters(
        cloudiness= 0,
        precipitation=0,
        sun_altitude_angle= 45,
        fog_density = 100,
        fog_distance = 0,
        fog_falloff = 0,
        )
        set_weather(world, weather)

        blueprint = world.get_blueprint_library()
        world_map = world.get_map()
        
        vehicle = add_vehicle(world, blueprint, vehicle_type='vehicle.audi.a2')
        global_vehicle = vehicle
        # Enables or disables the simulation of physics on this actor.
        vehicle.set_simulate_physics(True)
        physics_control = vehicle.get_physics_control()
        max_steer_angle = np.deg2rad(physics_control.wheels[0].max_steer_angle)


        spawn_points = world_map.get_spawn_points()
        waypoint_tuple_list = world_map.get_topology()
        origin_map = get_map(waypoint_tuple_list)

        agent = BasicAgent(vehicle, target_speed = 30)

        # prepare map
        destination = carla.Transform()
        destination.location = world.get_random_location_from_navigation()

    finally:

        if original_settings:
            world.apply_settings(original_settings)

        if (world and world.recording_enabled):
            client.stop_recorder()

        if world is not None:
            world.destroy()


if __name__ == '__main__':
    main()