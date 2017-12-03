import numpy as np
import sys

# Reference: https://en.wikipedia.org/wiki/Activation_function

class Sigmoid():
    def __init__(self): pass 

    def function(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.function(x) * (1 - self.function(x))

class Softmax():
    def __init__(self): pass 

    def function(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def gradient(self, x):
        p = self.function(x)
        return p * (1 - p)

class TanH():
    def __init__(self): pass

    def function(self, x):
        # Avoid overflow for large inputs
        x = np.clip(-100, 100, x)
        return 2 / (1 + np.exp(-2*x)) - 1

    def gradient(self, x):
        # Avoid overflow for large inputs
        #x = np.clip(-100, 100, x)
        return 1 - np.power(self.function(x), 2)

class ReLU():
    def __init__(self): pass 

    def function(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)

class LeakyReLU():
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def function(self, x):
        return np.where(x >= 0, x, self.alpha * x)

    def gradient(self, x):
        return np.where(x >= 0, 1, self.alpha)

class ELU():
    def __init__(self, alpha=0.1):
        self.alpha = alpha 

    def function(self, x):
        return np.where(x >= 0.0, x, self.alpha * (np.exp(x) - 1))

    def gradient(self, x):
        return np.where(x >= 0.0, 1, self.function(x) + self.alpha)

class SELU():
    # Reference :   https://arxiv.org/abs/1706.02515
    def __init__(self):
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946 

    def function(self, x):
        return self.scale * np.where(x >= 0.0, x, self.alpha*(np.exp(x)-1))

    def gradient(self, x):
        return self.scale * np.where(x >= 0.0, 1, self.alpha * np.exp(x))

class SoftPlus():
    def __init__(self): pass 

    def function(self, x):
        return np.log(1 + np.exp(x))

    def gradient(self, x):
        return Sigmoid().function(x)
