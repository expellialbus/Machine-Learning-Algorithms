import os 

def get_parent_dir(n=1):
    current_path = os.path.abspath(__file__)

    for _ in range(n):
        current_path = os.path.dirname(current_path)

    return current_path

os.sys.path.append(get_parent_dir())

from BatchGradientDescent import BatchGradientDescent
from StochasticGradientDescent import StochasticGradientDescent