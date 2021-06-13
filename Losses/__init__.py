import os

def get_parent_dir(n=0):
    current_path = os.path.dirname(os.path.abspath(__file__))

    for _ in range(n):
        current_path = os.path.dirname(current_path)

    return current_path

os.sys.path.append(get_parent_dir())

from Loss import Loss
from MeanSquaredError import MeanSquaredError
from MeanAbsoluteError import MeanAbsoluteError
from MeanAbsolutePercentageError import MeanAbsolutePercentageError
from MeanSquaredLogarithmicError import MeanSquaredLogarithmicError
from CosineSimilarity import CosineSimilarity
from HuberLoss import HuberLoss