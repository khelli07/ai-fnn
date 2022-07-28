# Artificial Neural Network / Feedforward Neural Network

Ah finally, a working code after hours of debugging. 🥴

I've made two articles that explains how this ANN/FNN work [here](https://khelli07.medium.com/introduction-to-artificial-neural-network-in-deep-learning-aa7ba2280f50). Kindly visit them, thanks in advance :)

## How to use

I have two version of code in src folder: with and without object oriented programming (OOP). I will explain the one with oop only. The folder is located in src/nn, so if you want to import, use src.nn.*

### There are several features you can use

- Activation: ReLU and Sigmoid
- Layer: Dense
- Loss: MeanSquaredError and BinaryCrossentropy (not stable yet)
- Metrics: Accuracy (for classification task only)
- Model: Sequential
- Optimizer: SGD (Stochastic Gradient Descent)

### Model usage

- To initialize model, simply do Sequential()
- To add layer, do model.add({your layer})
- To compile, do model.compile({params}). Compiling will define your loss function, metrics, and optimizer.
- Finally, you can fit and predict your data.

### Example

To see an example, you can see main.ipynb or main_test.py file.


If you are interested more in deep learning, I suggest you to see [this book.](http://neuralnetworksanddeeplearning.com/) 