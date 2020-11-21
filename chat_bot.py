import random
import json
import torch
from chatbot_model import NeuralNetwork
from nltk_utility import tokenize, bag_of_words

def setup():
    device = torch.device('cpu')

    with open('intents.json', 'r') as file:
        intents = json.load(file)

    FILE = "model.pth"
    model_data = torch.load(FILE)

    input_size = model_data['input_size']
    hidden_size = model_data['hidden_size']
    output_size = model_data['output_size']

    all_words = model_data['all_words']
    tags = model_data['tags']
    model_state = model_data['model_state']

    model = NeuralNetwork(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(model_state)
    # Set model to evaluation mode
    model.eval()
    return model, all_words, tags

class ChatBot():
    def __init__(self, model, name, all_words, tags):
        self.model = model
        self.name = name
        self.all_words = all_words
        self.tags = tags
        self.prompt = 'Bot: '
        self.greeting = 'Welcome to GroundsKeeper Coffee, Co.! My name is ' + self.name + '.'
    
    def greet(self):
        print(self.greeting)
        print('Enter \'done\' to exit.')

    def get_input(self):
        user_in = input('You: ')
        if (user_in.lower() == 'done'):
            print(self.prompt, 'Thanks for chatting! Have a great day!')
            exit(1)
        return user_in

    def evaluate_input(self, user_in):
        tok_input = tokenize(user_in)
        bag = bag_of_words(tok_input, self.all_words)
        #TODO: What is going on here?
        bag = bag.reshape(1, bag.shape[0])
        bag = torch.from_numpy(bag)

        output = self.model(bag)
        _, predicted = torch.max(output, dim=1)
        intent = self.tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        intent_prob = probs[0][predicted.item()]
        if intent_prob > 0.75:
            print(self.prompt, intent)
            print(float(intent_prob))
        else:
            print(self.prompt, 'Sorry, I didn\'t catch that. Can you say that again?')


def chat():
    model, all_words, tags = setup()
    chatbot = ChatBot(model, 'Kyle', all_words, tags)
    chatbot.greet()
    user_input = chatbot.get_input()
    chatbot.evaluate_input(user_input)

def main():
    chat()

main()

        


