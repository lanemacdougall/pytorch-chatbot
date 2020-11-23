import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_labels):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.layer3 = nn.Linear(hidden_size2, num_labels)
        self.relu = nn.ReLU()

    def forward(self, input):
        # Layer 1
        output = self.layer1(input)
        output = self.relu(output)
        # Layer 2
        output = self.layer2(output)
        output = self.relu(output)
        # Layer 3
        output = self.layer3(output)
        # Do not apply activation function or softmax after layer 3 because we are going to apply the cross-entropy loss function in training.py
        return output





