import json
import nltk_utility
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from chatbot_model import NeuralNetwork
from os.path import realpath as path

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

    punct = ['.', ',', ';', ':', '?', '!', '/', '[', ']', '{', '}', '(', ')']
    stop_words  = ['ourselves', 'between', 'but', 'again', 'there', 'once', 'during', 'out', 'very', 'with', 'they', 'an', 'be', 'some', 'for', 'its', 'such', 'into', 'of', 'most', 'itself',
                'other', 'off', 'is', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'his', 'through', 'don\'t', 'won\'t', 'nor', 'were', 'her',
                'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'in', 'will', 
                'on', 'does', 'then', 'that', 'because', 'over', 'so', 'did', 'under', 'he', 'herself', 'just', 'too', 'only', 'myself', 'which', 'those', 'after', 'few', 'whom', 'being', 'if', 'theirs',
                'against', 'a', 'by', 'doing', 'further', 'was', 'here', 'than', 'well', 'cannot', 'can\'t', 'found', 'would', 'about', 'own']
    all_words = [nltk_utility.stem(word) for word in all_words if word not in punct and word.lower() not in stop_words]

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

    return input_data, labels, tags, all_words


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
    

def train(input_data, labels, tags, all_words):
    # Training hyperparameters:
    batch_size = 8
    input_size = len(input_data[0])
    hidden_size1 = 64
    hidden_size2 = 32
    output_size = len(tags)
    learning_rate = 0.001
    epochs = 2500

    # DataLoader parameters
    shuffle_arg = True
    num_workers_arg = 0

    device = torch.device('cpu')

    data_set = TrainingData(input_data, labels)
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=shuffle_arg, num_workers=num_workers_arg)

    model = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size).to(device)

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
        

        if (epoch + 1) % 100 == 0:
            print('Epoch:', (epoch+1), 'Loss: ', loss.item())
    
    model_data = {
        "model_state" : model.state_dict(),
        "input_size" : input_size,
        "output_size" : output_size,
        "hidden_size1" : hidden_size1,
        "hidden_size2" : hidden_size2,
        "all_words" : all_words,
        "tags" : tags,
    }

    FILE = "model.pth"
    torch.save(model_data, FILE)
    print("Training complete. Model data saved to", path(FILE))



def main():
    input_data, labels, tags, all_words = preprocess()
    train(input_data, labels, tags, all_words)

main()
