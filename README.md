# Digit_Classifier_MNIST
Convolutional neural network for the classification of hand-written digits (0-9) trained on TensorFlow's MNIST dataset.

* The model was built using the Sequential class from TensorFlow's Keras API. 
* The initial layer is Conv2D with a 3x3 kernel and 8 filters. Rectifier activation function was used.
* Max pooling was done with a 2x2 window, and default strides.
* A flaten layer was added to reduce the input into a one-dimensional tensor.
* The model has two dense hidden layers with 64 units each and rectifier activation functions.
* The dense output layer has 10 units and uses the softmax activation function.

The final accuracy obtained after training for 5 epochs was <ins>**95.66%**</ins>.
