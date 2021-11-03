from importlib import import_module
import os

def get_parent_dir(n=0):
    current_path = os.path.dirname(os.path.abspath(__file__))

    for _ in range(n):
        current_path = os.path.dirname(current_path)

    return current_path

def get_optimizer(name):
    module = import_module(name)
    optimizer = getattr(module, name)

    return optimizer()  # creates an instance of the class

os.sys.path.append(get_parent_dir())

from Optimizer import Optimizer
from BatchGradientDescent import BatchGradientDescent
from StochasticGradientDescent import StochasticGradientDescent
from MiniBatchGradientDescent import MiniBatchGradientDescent
from Momentum import Momentum
from NesterovAcceleratedGradient import NesterovAcceleratedGradient
from AdaGrad import AdaGrad
from RMSprop import RMSprop
from Adadelta import Adadelta
from Adam import Adam
from AdaMax import AdaMax
from Nadam import Nadam
from AMSGrad import AMSGrad