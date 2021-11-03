from importlib import import_module
import os

def get_parent_dir(n=0):
    current_path = os.path.dirname(os.path.abspath(__file__))

    for _ in range(n):
        current_path = os.path.dirname(current_path)

    return current_path

def get_activation(name):
    module = import_module(name)
    activation = getattr(module, name)

    return activation()     # creates an instance of the class

os.sys.path.append(get_parent_dir())

from Activation import Activation
from Relu import Relu