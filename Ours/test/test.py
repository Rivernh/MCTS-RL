"""
Created on Mon Dec 07 16:34:10 2015

@author: SuperWang
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import itertools
import gym
from gym import spaces
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import functional as F
import random
from scipy.stats import norm
#a = torch.ones(125).reshape(5,5,5)
#a = F.softmax(a)

#print(a)

#b = math.cos(90/180*math.pi)
#print(b)

"""
a = torch.randn(1, 5)

m = nn.LayerNorm(a.size()[1:], elementwise_affine= False)

b = F.softmax(m(a))
print(a,b)


a = [[1,1],[1,0]]
a = np.array([a,a]).reshape((-1,2,2))
a = torch.tensor(torch.from_numpy(a)).cuda().float()
print(a)
b = F.softmax(a,dim=-1)
a = F.log_softmax(a,dim=-1)
print(a)
print(b)
a = list(itertools.product(np.array([ 2.5, 5, 7.5, 10]), np.array([-0.3,-0.15, 0, 0.15, 0.3])))
a.insert(0,(0,0))
print(a)


a = [[0.9,0.4],[0.5,0.8]]
a = np.array([a,a]).reshape((-1,2,2))
a = torch.tensor(torch.from_numpy(a)).cuda().float()
print(a)
b = F.softmax(a,dim=0)
a = F.log_softmax(a,dim=1)
a = np.exp(a.data.cpu().numpy().flatten())

print(b)
print(sum(a))"""
x = [-0.4,-0.2,0,0.2,0.4]
a = norm.pdf(x,0,0.25)
print(np.pi)