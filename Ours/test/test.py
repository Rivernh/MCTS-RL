"""
Created on Mon Dec 07 16:34:10 2015

@author: SuperWang
"""

import matplotlib.pyplot as plt
import numpy as np
import itertools
import gym
from gym import spaces
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F

a = torch.ones(125).reshape(5,5,5)
a = F.softmax(a)

print(a)