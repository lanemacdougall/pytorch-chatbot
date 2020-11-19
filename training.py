import json
import nltk_utility
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from chatbot_model import NeuralNetwork

def preprocess():
    with open("intents.json", "r") as file:
        intents = json.load(file)

    all_words = []
    tags = []
    xy_data = []

    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            tok_pattern = nltk_utility.tokenize(pattern)
            all_words.extend(tok_pattern)
            xy_data.append((tok_pattern, tag))

    ignore = ['.', ',', ';', ':', '?', '!', '/', '[', ']', '{', '}', '(', ')']
    all_words = [nltk_utility.stem(word) for word in all_words if word not in ignore]

    #print(all_words)
    all_words = sorted(set(all_words))
    tags = sorted(tags)

    # Initialized as arrays because it is easier to append to arrays than to numpy arrays; but will need to convert to numpy arrays later
    input_data = []
    labels = []

    for (tok_pattern, tag) in xy_data:
        bag_of_words = nltk_utility.bag_of_words(tok_pattern, all_words)
        input_data.append(bag_of_words)
        labels.append(tags.index(tag)) # Model is trained using cross-entropy loss function and therefore we do not want labels to be one-hot encoded
    
    # Using PyTorch, so we need to convert arrays into numpy arrays
    input_data = np.array(input_data)
    labels = np.array(labels)

    return input_data, labels, tags


# Create PyTorch dataset with training data

class TrainingData(Dataset):
    def __init__(self, input_data, labels):
        super(TrainingData, self).__init__()
        self.num_samples = len(input_data)
        self.input_data = input_data
        self.labels = labels
    
    def __getitem__(self, index):
        return (self.input_data[index], self.labels[index])

    def __len__(self):
        return self.num_samples
    

def train(input_data, labels, tags):
    # Training hyperparameters:
    batch_size = 8
    input_size = len(input_data[0])
    hidden_size = 8
    output_size = len(tags)
    learning_rate = 0.001
    epochs = 2000


    device = torch.device('cpu')

    data_set = TrainingData(input_data, labels)
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=True, num_workers=0)

    #data_loader_iter = iter(data_loader)
    #for (words, labels) in data_loader_iter:
    #    print(words, labels)

    model = NeuralNetwork(input_size, hidden_size, output_size).to(device)

    # Loss function and optimization
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for (words, labels) in data_loader:
            words = words.to(device)
            labels = labels.to(device)

            # Forward feed
            output = model(words)
            
            # Calculate loss
            loss = criterion(output, labels)

            # Backpropagation and optimization
            # Empty gradients
            optimizer.zero_grad()
            # Calculate backpropagation
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(epoch+1, epochs, loss.item())


def main():
    input_data, labels, tags = preprocess()
    train(input_data, labels, tags)

main()
