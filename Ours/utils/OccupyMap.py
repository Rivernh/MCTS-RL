import numpy as np

class Map:
    def __init__(self):
        self.size = [40,7]
        self.step = 2 # m
        self.map = np.zeros(self.size)

    def update(self,carla_env):
        pass




if __name__ == '__main__':
    a = Map()
    print(a.map)