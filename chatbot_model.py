import torch
import torch.nn as nn

# Feed-forward neural network used to classify the intent of English sentences

class NeuralNetwork(nn.Module):   # Class inherits from torch.nn's Module class
    def __init__(self, input_size, hidden_size1, hidden_size2, num_labels):
        super(NeuralNetwork, self).__init__()
        # Input layer takes input of input size, outputs size of hidden_layer1
        self.layer1 = nn.Linear(input_size, hidden_size1)
        # Hidden layer takes input of hidden_layer1, outputs size of hidden_layer2
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        # Output layer takes input of hidden_layer2, outputs size of the number of labels (classes) 
        self.layer3 = nn.Linear(hidden_size2, num_labels)
        # Using rectified linear (ReLU) activation function in network
        self.relu = nn.ReLU()

    # Function driving forward pass through network (i.e., does not include the backpropagation that occurs in training)
    def forward(self, input):
        # For each layer (excluding output layer), pass data into layer and then call the relu (ReLU) method on the output of the layer
        # Layer 1
        output = self.layer1(input)
        output = self.relu(output)
        # Layer 2
        output = self.layer2(output)
        output = self.relu(output)
        # Do not apply activation function or softmax on output of layer 3 because we are going to apply the cross-entropy loss function on the output during training
        # Layer 3
        output = self.layer3(output)
        return output





