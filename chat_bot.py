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
    hidden_size1 = model_data['hidden_size1']
    hidden_size2 = model_data['hidden_size2']
    output_size = model_data['output_size']

    all_words = model_data['all_words']
    tags = model_data['tags']
    model_state = model_data['model_state']

    model = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size).to(device)
    model.load_state_dict(model_state)
    # Set model to evaluation mode
    model.eval()
    return model, all_words, tags, intents

class ChatBot():
    def __init__(self, model, name, all_words, tags):
        self.model = model
        self.name = name
        self.all_words = all_words
        self.tags = tags
        self.prompt = 'Bot: '
        self.greeting = 'Welcome to GroundsKeeper Coffee, Co.! My name is ' + self.name + '.'
        self.context = ''
    
    def greet(self):
        print(self.greeting)
        print('Enter \'done\' to exit.')

    def get_input(self):
        user_in = input('You: ')
        return user_in

    def evaluate_input(self, user_in):
        tok_input = tokenize(user_in)
        bag = bag_of_words(tok_input, self.all_words)
        #TODO: What is going on here?
        bag = bag.reshape(1, bag.shape[0])
        bag = torch.from_numpy(bag)

        output = self.model(bag)
        #print(output)
        #for result in output:
        #    print(result)
        _, predicted = torch.max(output, dim=1)
        #print(torch.max(output, dim=1))

        intent = self.tags[predicted.item()]
        
        probs = torch.softmax(output, dim=1)
        #predictions = [result in output if probs[0][result.item()] > 0.75]
        #print(predictions)

        intent_prob = probs[0][predicted.item()]
        print(intent)
        print(float(intent_prob))
        if intent_prob > 0.75:
            return intent
        else:
            return 'fail'


def chat():
    model, all_words, tags, intents = setup()
    chatbot = ChatBot(model, 'Kyle', all_words, tags)
    chatbot.greet()
    states = [0] * len(tags)
    while True:
        user_input = chatbot.get_input()
        if (user_input.lower() == 'done'):
            print(chatbot.prompt, 'Thanks for chatting! Have a great day!')
            break
        tag = chatbot.evaluate_input(user_input)
        if tag == 'fail':
            print(chatbot.prompt, 'Sorry, I didn\'t catch that. Can you say that again?')
        else:
            for intent in intents['intents']:
                if tag == intent['tag']:
                    if tag == 'bot_name':
                        state = states[tags.index('bot_name')]
                        if state == 0:
                            print(chatbot.prompt, 'My name is ', chatbot.name, '! I will be your virtual assistant today here at GroundsKeeper Coffee, Co.', sep='')
                        elif state == 1:
                            print(chatbot.prompt, 'Sorry, I thought that I already told you! My name is ', chatbot.name, '! I will be your virtual assistant today here at GroundsKeeper Coffee, Co.', sep='')
                        else:
                            print(chatbot.prompt, 'I\'ve already told you, remember? I\'m ', chatbot.name, '!', sep='')
                        states[tags.index('bot_name')] += 1
                    else:
                        print(chatbot.prompt, intent['responses'][states[tags.index(tag)]])
                        if states[tags.index(tag)] < len(intent['responses'])-1:
                            states[tags.index(tag)] += 1
                    break



def main():
    chat()

main()

        


