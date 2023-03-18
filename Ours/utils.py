import numpy as np
import itertools
import random
def int2action(int):
    availables = list(itertools.product(np.array([0, 2.5, 5, 7.5, 10]), np.array([0, -0.1, -0.3, 0.1, 0.3])))
    print(availables[int])

if __name__ == '__main__':
    a =np.array([[1.0400253252246512e-05, 1.0400253252246512e-05], [1.4871155887133116e-05, 1.4871155887133116e-05], [2.0468583507894717e-05, 2.0468583507894717e-05]])
    print(a[:,0])
    print(random.randint(0,255))