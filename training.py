import json
import nltk_utility
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from chatbot_model import NeuralNetwork
from os.path import realpath as path

# Script carries out data pre-processing and model training

# Data Pre-Processing
def preprocess():

    # Data stored in intents.json - stored as a JSON object
    with open("intents.json", "r") as file:
        intents = json.load(file)

    all_words = []
    tags = []
    xy_data = []

    for intent in intents['intents']:
        # Retrieve and store intent's tag (label)
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            # For each pattern (training user inputs) in the intent, tokenize the pattern, add the pattern to the all_words array, 
            # and then add the pattern and its corresponding tag to the xy_data array (x referring to the patterns and y referring to the tag (label))
            tok_pattern = nltk_utility.tokenize(pattern)
            all_words.extend(tok_pattern)
            xy_data.append((tok_pattern, tag))

    # Ignore punctuation and special characters
    punct = ['.', ',', ';', ':', '?', '!', '/', '[', ']', '{', '}', '(', ')', '$', '@', '%', '&', '*', '#', '^']

    # Exclude stop words (words without meaning in this context) from consideration; stop words are thus unable to influence the network's predictions 
    stop_words  = ['ourselves', 'between', 'but', 'again', 'there', 'once', 'during', 'out', 'very', 'they', 'an', 'be', 'some', 'for', 'its', 'such', 'into', 'of', 'most', 'itself',
                'other', 'off', 'is', 'am', 'or', 'who', 'as', 'him', 'each', 'the', 'themselves', 'until', 'below', 'we', 'these', 'his', 'through', 'don\'t', 'won\'t', 'nor', 'were', 'her',
                'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'she', 'all', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'in', 'will', 
                'on', 'does', 'then', 'that', 'because', 'over', 'so', 'did', 'under', 'he', 'herself', 'just', 'too', 'only', 'myself', 'which', 'those', 'after', 'few', 'whom', 'being', 'if', 'theirs',
                'against', 'a', 'by', 'doing', 'further', 'was', 'here', 'than', 'well', 'cannot', 'can\'t', 'found', 'would', 'own', 'can', 'what', 'give', 'given', 'nice', 'not', 'also']
    
    # Remove punctuations, special characters, and stop words from all_words
    # Stem those words that are kept in all_words 
    all_words = [nltk_utility.stem(word) for word in all_words if word not in punct and word.lower() not in stop_words]
    # Convert all_words into a set (remove duplicates) and sort (lexicographically)
    all_words = sorted(set(all_words))
    # Sort tags array lexicographically; tags array does not contain any duplicates 
    tags = sorted(tags)

    # Initialize input_data and labels as standard Python arrays because it is easier to append to arrays than to numpy arrays; but will need to convert to numpy arrays later
    input_data = []
    labels = []
    for (tok_pattern, tag) in xy_data:
        # Generate a bag of words for each pattern in xy_data and add this bad of words to the input_data array
        bag_of_words = nltk_utility.bag_of_words(tok_pattern, all_words)
        input_data.append(bag_of_words)
        # Add the index of each tag (0 ... n, where n is the number of tags minus one) in tags to the labels array
        labels.append(tags.index(tag)) # Model is trained using cross-entropy loss function and therefore we do NOT want labels to be one-hot encoded
    # Using PyTorch, so we need to convert arrays into numpy arrays
    input_data = np.array(input_data)
    labels = np.array(labels)
    return input_data, labels, tags, all_words  # tags and all_words are needed in the train method


# Create PyTorch dataset with training data
# TrainingData object is used by DataLoader object in feeding data to neural network during training
class TrainingData(Dataset):  # Class inherits from PyTorch's Dataset class
    def __init__(self, input_data, labels):
        super(TrainingData, self).__init__()
        self.num_samples = len(input_data)
        self.input_data = input_data
        self.labels = labels
    
    def __getitem__(self, index):
        return (self.input_data[index], self.labels[index])

    def __len__(self):
        return self.num_samples
    

# Method carries out model training process 
def train(input_data, labels, tags, all_words):
    # Training hyperparameters:
    batch_size = 8  # Selected based on research; experimented with sizes 10, 16, and 20, but experienced best results with 8
    input_size = len(input_data[0])
    hidden_size1 = 64  # Selected based on recommendation that first hidden layer be a power of 2 and approx. half of the size of the input data
    hidden_size2 = 32  # Same thought process described on the above line influenced the selection of this layer size (power of 2 and about half the size of previous layer)
    output_size = len(tags)
    learning_rate = 0.001  # Selected based on research; rate most commonly utilized by similar applications
    epochs = 3000  # Largest number of epochs such that meaningful changes in the loss still occur towards the end of the training - little change in loss after 3000 epochs   

    # DataLoader parameters
    shuffle_arg = True  # Shuffle data after each batch
    num_workers_arg = 0  # Multi-processing is not used; any number greater than 0 raised an error during training

    device = torch.device('cpu')  # GPU support is not available on the device used to develop this program - switch to if-else statement if 'cuda' support may be available

    data_set = TrainingData(input_data, labels)
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=shuffle_arg, num_workers=num_workers_arg)

    # Instantiate model with hyperparameters and send to device
    model = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size).to(device)

    # Utilize cross-entropy loss function
    criterion = nn.CrossEntropyLoss()
    #Utilize Adam optimizer - selected based on research; Adam is commonly used by similar applications
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training run
    for epoch in range(epochs):
        for (words, labels) in data_loader:

            # Send patterns and labels loaded by DataLoader to device
            words = words.to(device)
            labels = labels.to(device)

            # Forward feed data through neural network and retrieve output of model
            output = model(words)
            
            # Calculate loss (cross-entropy loss)
            loss = criterion(output, labels)

            # Backpropagation and optimization

            # Empty gradients
            optimizer.zero_grad()
            # Calculate backpropagation
            loss.backward()
            # Update parameters after gradients are computed 
            optimizer.step()
        
        # Show details of training (epoch and loss) every 100 epochs
        if (epoch + 1) % 100 == 0:
            print('Epoch:', (epoch+1), 'Loss: ', loss.item())
    
    # Save model data for later use
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

# Run data pre-processing and model training
def main():
    input_data, labels, tags, all_words = preprocess()
    train(input_data, labels, tags, all_words)

main()
