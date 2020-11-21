import random
import json
import torch
from chatbot_model import NeuralNetwork
from nltk_utility import tokenize, bag_of_words

def setup():
    device = torch.device('cpu')

    '''
    with open('intents.json', 'r') as file:
        intents = json.load(file)
    '''

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
            #print(float(intent_prob))
            return intent
        else:
            return 'fail'


def chat():
    model, all_words, tags = setup()
    chatbot = ChatBot(model, 'Kyle', all_words, tags)
    chatbot.greet()
    while True:
        user_input = chatbot.get_input()
        if (user_input.lower() == 'done'):
            print(chatbot.prompt, 'Thanks for chatting! Have a great day!')
            break
        intent = chatbot.evaluate_input(user_input)
        if intent == 'fail':
            print(chatbot.prompt, 'Sorry, I didn\'t catch that. Can you say that again?')
        elif intent == 'greeting':
            print(chatbot.prompt, 'Hello! How can I be of assistance?')
        elif intent == 'goodbye':
            print(chatbot.prompt, 'Bye, come back again soon.')
        elif intent == 'thanks':
            print(chatbot.prompt, 'Happy to help!')
        elif intent == 'how_are_you':
            print(chatbot.prompt, 'I am great, thank you!')
        elif intent == 'bot_name':
            print(chatbot.prompt, 'My name is ', chatbot.name, '! I will be your virtual assistant today here at GroundsKeeper Coffee, Co.', sep='')
        elif intent == 'options':
            print(chatbot.prompt, 'I am the virtual spokesperson for GroundsKeeper Coffee Supply, Co.!\nI am here to answer any of your questions concerning our business.\nI am also able to answer your general coffee-related queries.')
        elif intent == 'coffee_supplies':
            print(chatbot.prompt, 'We offer large-scale coffee bean supplies in a variety of coffee bean types.\nWe primarily sell Aribica beans grown in Central and South America; however, we also sell a few Ethiopian coffee species.')
        elif intent == 'payment':
            print(chatbot.prompt, 'We accept cash, personal checks, credit, PayPal, and CashApp.')
        elif intent == 'shipping_time':
            print(chatbot.prompt, 'Shipping will take approximately 2 - 6 days, depending on your location.')
        elif intent == 'shipping_cost':
            print(chatbot.prompt, 'Shipping cost depends entirely on the size of the shipment and your location.')
        elif intent == 'business_hours':
            print(chatbot.prompt, 'We are open from 5 AM to 7 PM, Monday through Friday. You may speak with a representative during those hours at 1-800-GROUNDS.')
        else:
            print(intent)


def main():
    chat()

main()

        


