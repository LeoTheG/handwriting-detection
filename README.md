# Pytorch Project

## Local system setup

1. Install pytorch with cuda (https://pytorch.org/get-started/locally/)
2. Check if installation successful: `python3 -c "import torch; print(torch.cuda.is_available())"` (should return True)

## Training

`python3 src/train_model.py`

Will create a model `mnist_model.pth`

## Running

`python3 src/main.py test_image_3.png`

Expects images to be 28x28 pixels, white on black background

## Explanation

Epoch:

An epoch refers to one complete forward and backward pass of all the training examples through the neural network.
In other words, an epoch is a single pass through the entire training dataset.
During training, we usually iterate over the training data multiple times. Each of these iterations is called an epoch.
For example, if you train a neural network for 10 epochs, it means you've passed the entire dataset through the neural network 10 times.
Why multiple epochs? As the model sees the data multiple times, it has a better chance of learning the patterns and making accurate predictions. However, too many epochs can also lead to overfitting, where the model performs well on the training data but poorly on unseen data.

Loss (or Loss Function or Cost Function):

The loss function quantifies how well or poorly a given prediction model is performing. It's a measure of the difference between the predicted values and the actual values.
In simple terms, it's a way of measuring "how wrong" the model's predictions are.
During training, the goal is typically to minimize this loss value. As the loss decreases, the model's predictions get better.
Different problems may have different loss functions. For instance:
Regression tasks might use Mean Squared Error (MSE).
Classification tasks might use Cross-Entropy Loss.
The process of training a neural network involves feeding data through the network, calculating the loss, and then updating the weights of the network to try and reduce this loss. This cycle repeats over multiple epochs.
