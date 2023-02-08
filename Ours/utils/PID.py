import numpy as np

class PID:
    def __init__(self, kp = 0.12 , ki = 0.02, kd = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.first = True
        self.last = 0.0
        self.sum = 0.0
        self.step = 0

    def reset(self):
        self.first = True
        self.sum = 0.0
        self.step = 0
    
    def run(self, target, now):
        self.step += 1
        if self.step > 10000:
            self.sum = 0
        error = target - now
        self.sum += error
        if self.first:
            self.last = error
            self.first = False
        output = self.kp * error + self.ki * self.sum + self.kd * (error - self.last)
        return output

class IncreasPID:
    def __init__(self, kp = 1.0, ki = 0.0, kd = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.first = True
        self.last = 0.0
        self.lastlast = 0.0

    def reset(self):
        self.first = True
    
    def run(self, target, now):
        error = target - now
        if self.first:
            self.last = error
            self.lastlast = error
            self.first = False
        output = self.kp * (error - self.last) + self.ki * error + self.kd * (error - 2 * self.last + self.lastlast)
        return output
