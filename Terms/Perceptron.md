# Perceptron
In machine learning, the `perceptron` (or `McCulloch–Pitts neuron`) is an algorithm for supervised learning of binary classifiers. A binary classifier is a function which can decide whether or not an input, represented by a vector of numbers, belongs to some specific class.  
It is a type of linear classifier, i.e. a classification algorithm that makes its predictions based on a linear predictor function combining a set of weights with the feature vector.

Rosenblatt described the details of the perceptron in a 1958 paper. His organization of a perceptron is constructed of three kinds of cells ("units"): AI, AII, R, which stand for `"projection"`, `"association"` and `"response"`.

**Key Components of a Perceptron:**  
**Input Features:** These are the characteristics or attributes of the input data.  
**Weights:** Each input feature is associated with a weight, which determines its significance in influencing the perceptron’s output.  
**Summation Function:** This function calculates the weighted sum of the inputs.  
**Activation Function:** The weighted sum is passed through an activation function, typically a step function, to produce the output.  
**Bias:** An additional parameter that allows the model to make adjustments independent of the input.

**Types of Perceptrons:**  
- **Single-Layer Perceptron:** Consists of a single layer of input nodes connected to output nodes. It can only learn linearly separable patterns.
- **Multilayer Perceptron:** Contains multiple layers, allowing it to handle more complex patterns and relationships within the data.  

Perceptrons are the simplest form of neural networks and have paved the way for more complex architectures in deep learning.