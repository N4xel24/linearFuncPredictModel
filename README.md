# fx_predict

A simple neural network project built with PyTorch.

This project trains a neural network to predict the output of a mathematical function from generated training data.

## why 67 hidden neurons?

idk

## Project structure
```text
.
├── models/
├── training_data/
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   └── training.py
├── test.py
└── README.md
```

## Neural network

* Input: 1 neuron
* Hidden layer 1: 67 neurons
* Hidden layer 2: 67 neurons
* Output: 1 neuron

Activation function: ReLU

Loss function: MSELoss

Optimizer: Adam

## Dataset

The current dataset contains 500 samples generated from the function:

y = 2x + 3

## Training

Run:

```
python src/training.py
```

The model with the lowest loss is saved automatically.

## Prediction

Run:

```
python src/predict.py
```

Then enter the function name and the input value.

## Requirements

* Python 3.12
* PyTorch
* Pandas

## Purpose

This project was created while PyTorch was still cooking my biological neural network.
