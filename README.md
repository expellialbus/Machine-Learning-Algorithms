# Project File Structrue

```
Machine-Learning-Algorithms/
├── Activations
│   ├── __pycache__
│   │   ├── Activation.cpython-38.pyc
│   │   ├── __init__.cpython-38.pyc
│   │   └── Relu.cpython-38.pyc
│   ├── Activation.py
│   ├── __init__.py
│   └── Relu.py
├── Losses
│   ├── __pycache__
│   │   ├── CosineSimilarity.cpython-38.pyc
│   │   ├── GeometricMeanAbsoluteError.cpython-38.pyc
│   │   ├── GeometricMeanRelativeAbsoluteError.cpython-38.pyc
│   │   ├── GeometricRootMeanSquaredError.cpython-38.pyc
│   │   ├── HuberLoss.cpython-38.pyc
│   │   ├── __init__.cpython-38.pyc
│   │   ├── LogCosh.cpython-38.pyc
│   │   ├── Loss.cpython-38.pyc
│   │   ├── MaximumAbsoluteError.cpython-38.pyc
│   │   ├── MeanAbsoluteError.cpython-38.pyc
│   │   ├── MeanAbsolutePercentageError.cpython-38.pyc
│   │   ├── MeanAbsoluteRelativeError.cpython-38.pyc
│   │   ├── MeanError.cpython-38.pyc
│   │   ├── MeanNormalizedBias.cpython-38.pyc
│   │   ├── MeanPercentageError.cpython-38.pyc
│   │   ├── MeanRelativeAbsoluteError.cpython-38.pyc
│   │   ├── MeanSquaredError.cpython-38.pyc
│   │   ├── MeanSquarePercentageError.cpython-38.pyc
│   │   ├── MedianAbsoluteError.cpython-38.pyc
│   │   ├── MedianAbsolutePercentageError.cpython-38.pyc
│   │   ├── MedianRelativeAbsoluteError.cpython-38.pyc
│   │   ├── MedianSquarePercentageError.cpython-38.pyc
│   │   ├── Quantile.cpython-38.pyc
│   │   ├── RelativeAbsoluteError.cpython-38.pyc
│   │   ├── RelativeSquaredError.cpython-38.pyc
│   │   ├── RootMeanSquaredError.cpython-38.pyc
│   │   ├── RootMeanSquarePercentageError.cpython-38.pyc
│   │   ├── RootMedianSquarePercentageError.cpython-38.pyc
│   │   ├── RootRelativeSquaredError.cpython-38.pyc
│   │   ├── SumOfAbsoluteDifference.cpython-38.pyc
│   │   ├── SumOfSquaredError.cpython-38.pyc
│   │   ├── SymmetricMeanAbsolutePercentageError.cpython-38.pyc
│   │   └── SymmetricMedianAbsolutePercentageError.cpython-38.pyc
│   ├── CosineSimilarity.py
│   ├── GeometricMeanAbsoluteError.py
│   ├── GeometricMeanRelativeAbsoluteError.py
│   ├── GeometricRootMeanSquaredError.py
│   ├── HuberLoss.py
│   ├── __init__.py
│   ├── LogCosh.py
│   ├── Loss.py
│   ├── MaximumAbsoluteError.py
│   ├── MeanAbsoluteError.py
│   ├── MeanAbsolutePercentageError.py
│   ├── MeanAbsoluteRelativeError.py
│   ├── MeanError.py
│   ├── MeanNormalizedBias.py
│   ├── MeanPercentageError.py
│   ├── MeanRelativeAbsoluteError.py
│   ├── MeanSquaredError.py
│   ├── MeanSquarePercentageError.py
│   ├── MedianAbsoluteError.py
│   ├── MedianAbsolutePercentageError.py
│   ├── MedianRelativeAbsoluteError.py
│   ├── MedianSquarePercentageError.py
│   ├── Quantile.py
│   ├── RelativeAbsoluteError.py
│   ├── RelativeSquaredError.py
│   ├── RootMeanSquaredError.py
│   ├── RootMeanSquarePercentageError.py
│   ├── RootMedianSquarePercentageError.py
│   ├── RootRelativeSquaredError.py
│   ├── SumOfAbsoluteDifference.py
│   ├── SumOfSquaredError.py
│   ├── SymmetricMeanAbsolutePercentageError.py
│   └── SymmetricMedianAbsolutePercentageError.py
├── Models
│   ├── LinearModels
│   │   ├── __pycache__
│   │   │   ├── __init__.cpython-38.pyc
│   │   │   ├── LinearRegression.cpython-38.pyc
│   │   │   └── LogisticRegression.cpython-38.pyc
│   │   ├── __init__.py
│   │   ├── LinearRegression.py
│   │   └── LogisticRegression.py
│   ├── NeuralNetworks
│   │   ├── Layers
│   │   │   ├── __pycache__
│   │   │   │   ├── Dense.cpython-38.pyc
│   │   │   │   ├── __init__.cpython-38.pyc
│   │   │   │   └── Layer.cpython-38.pyc
│   │   │   ├── Dense.py
│   │   │   ├── __init__.py
│   │   │   └── Layer.py
│   │   ├── __pycache__
│   │   │   └── __init__.cpython-38.pyc
│   │   └── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-38.pyc
│   │   └── Model.cpython-38.pyc
│   ├── __init__.py
│   └── Model.py
├── Optimizers
│   ├── __pycache__
│   │   ├── Adadelta.cpython-38.pyc
│   │   ├── AdaGrad.cpython-38.pyc
│   │   ├── AdaMax.cpython-38.pyc
│   │   ├── Adam.cpython-38.pyc
│   │   ├── AMSGrad.cpython-38.pyc
│   │   ├── BatchGradientDescent.cpython-38.pyc
│   │   ├── __init__.cpython-38.pyc
│   │   ├── MiniBatchGradientDescent.cpython-38.pyc
│   │   ├── Momentum.cpython-38.pyc
│   │   ├── Nadam.cpython-38.pyc
│   │   ├── NesterovAcceleratedGradient.cpython-38.pyc
│   │   ├── Optimizer.cpython-38.pyc
│   │   ├── RMSprop.cpython-38.pyc
│   │   └── StochasticGradientDescent.cpython-38.pyc
│   ├── Adadelta.py
│   ├── AdaGrad.py
│   ├── AdaMax.py
│   ├── Adam.py
│   ├── AMSGrad.py
│   ├── BatchGradientDescent.py
│   ├── __init__.py
│   ├── MiniBatchGradientDescent.py
│   ├── Momentum.py
│   ├── Nadam.py
│   ├── NesterovAcceleratedGradient.py
│   ├── Optimizer.py
│   ├── RMSprop.py
│   └── StochasticGradientDescent.py
├── __init__.py
├── LICENSE
├── main.py
└── README.md

14 directories, 122 files
```
