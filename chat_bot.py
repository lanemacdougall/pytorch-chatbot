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
        self.abilities = 'I am here to answer questions concerning our business\' products, our hours of operation, the availability of our customer service representatives, shipping, and payment.\n\
Also, whenever you\'re ready, I am here to take your order. So just let me know when you\'re ready to see the varieties of coffee we currently offer!\nIf you ever forget these options and want a reminder, just let me know!'
        self.user_context = ''
    
    def greet(self):
        print("\n" + self.greeting, end="\n\n")
        print(self.abilities, end="\n\n")
        print('Enter \'done\' to exit.\n')

    def get_input(self):
        user_in = input('You: ')
        return user_in

    def evaluate_input(self, user_in):
        tok_input = tokenize(user_in)
        bag = bag_of_words(tok_input, self.all_words)
        
        #Reshape numpy array to match shape of neural network's expected input
        bag = bag.reshape(1, bag.shape[0])
        bag = torch.from_numpy(bag)

        output = self.model(bag)
        _, predicted = torch.max(output, dim=1)

        intent = self.tags[predicted.item()]
        
        probs = torch.softmax(output, dim=1)

        intent_prob = probs[0][predicted.item()]
        print(intent)
        print(float(intent_prob))
        if intent_prob > 0.75:
            return intent
        else:
            return 'fail'


def chat():
    model, all_words, tags, intents = setup()
    chatbot = ChatBot(model, 'Bean', all_words, tags)
    chatbot.greet()
    states = [0] * len(tags)
    count = 0
    selected_varieties = []
    selected_weights = []
    while True:
        user_input = chatbot.get_input()
        if user_input.lower() == 'done':
            print(chatbot.prompt, 'Thanks for chatting! Have a great day!')
            break
        tag = chatbot.evaluate_input(user_input)
        
        if chatbot.user_context == 'select_variety':
            if 'no' in user_input.lower() or 'not' in user_input.lower() or 'nah' in user_input.lower() or 'pass' in user_input.lower() or 'nevermind' in user_input.lower() or ('changed' in user_input.lower() and 'mind' in user_input.lower()):
                print(chatbot.prompt, 'Okay! How else can I assist you?')
                chatbot.user_context = ''
                continue
            elif 'ethiopian' in user_input.lower() and not 'colombian' in user_input.lower() and not 'guatemalan' in user_input.lower():
                selected_varieties.append('Ethiopian')
            elif not 'ethiopian' in user_input.lower() and 'colombian' in user_input.lower() and not 'guatemalan' in user_input.lower():
                selected_varieties.append('Colombian')
            elif not 'ethiopian' in user_input.lower() and not 'colombian' in user_input.lower() and 'guatemalan' in user_input.lower():
                selected_varieties.append('Guatemalan')
            elif 'ethiopian' in user_input.lower() and 'colombian' in user_input.lower() and not 'guatemalan' in user_input.lower():
                selected_varieties.append('Ethiopian')
                selected_varieties.append('Colombian')
            elif 'ethiopian' in user_input.lower() and not 'colombian' in user_input.lower() and 'guatemalan' in user_input.lower():
                selected_varieties.append('Ethiopian')
                selected_varieties.append('Guatemalan')
            elif not 'ethiopian' in user_input.lower() and 'colombian' in user_input.lower() and 'guatemalan' in user_input.lower():
                selected_varieties.append('Colombian')
                selected_varieties.append('Guatemalan')
            elif ('ethiopian' in user_input.lower() and 'colombian' in user_input.lower() and 'guatemalan' in user_input.lower()) or ("all three" in user_input.lower()):
                selected_varieties.append('Ethiopian')
                selected_varieties.append('Colombian')
                selected_varieties.append('Guatemalan')
            else:
                print(chatbot.prompt, 'I\'m sorry we only offer Ethiopian, Colombian, and Guatemalan varieties at this time. Please try again.')

            if len(selected_varieties) == 1:
                print(chatbot.prompt, 'Perfect. And what will be the weight (in pounds)?')
                chatbot.user_context = 'select_weight'
            elif len(selected_varieties) > 1:
                print(chatbot.prompt, 'Perfect. Please enter the weights (in pounds) for the ', end="")
                for i in selected_varieties:
                    print(i, end=", ")
                print("in that order and separated by a comma and a space.")
                chatbot.user_context = 'select_weight'
        elif chatbot.user_context == 'select_weight':
            if 'nevermind' in user_input.lower() or ('changed' in user_input.lower() and 'mind' in user_input.lower()):
                print(chatbot.prompt, 'Okay! How else can I assist you?')
                chatbot.user_context = ''
            else:
                for word in user_input.lower().split():
                    num = word.replace(",", "", 1)
                    altered_num = num.replace(".", "", 1)
                    if altered_num.isdigit():
                        selected_weights.append(num)
                if len(selected_weights) == 0:
                    print(chatbot.prompt, 'I\'m sorry, I\'m not seeing any valid weights. Please try again. Remember to list multiple weights separated by a comma and a space.')
                elif len(selected_weights) != len(selected_varieties):
                    print(chatbot.prompt, 'Please enter a weight for each of the coffee varieties that you selected for your order. Remember to list multiple weights separated by a comma and a space.')
                    selected_weights = []
                else:
                    order = dict(zip(selected_varieties, selected_weights))
                    print(chatbot.prompt, "Okay, fantastic. I have you put down for the following order:", end=" ")
                    count = 0
                    for variety in order:
                        if len(order) == 1:
                            print(order[variety] + " pounds of " + variety + ".")
                        else:
                            if count < len(order) - 1:
                                print(order[variety] + " pounds of " + variety, end=" ")
                            else:
                                print(' and ' + order[variety] + " pounds of " + variety + ".")
                            count += 1
                    print('Please feel free to proceed to checkout, or stay and ask me more questions!')
                    chatbot.user_context = ''
        else:
            if tag == 'fail' or user_input == '':
                print(chatbot.prompt, 'Sorry, I didn\'t catch that. Your question or comment may be outside of my scope of understanding.\nIf you had a typing error or are able to re-phrase your question or comment, please try again. Otherwise, please feel free to contact my developer with ways in which you believe I can be improved!')
            else:
                for intent in intents['intents']:
                    # Priority given to intents linked to current context
                    if 'context_filter' in intent and chatbot.user_context == intent['context_filter']:
                        if tag == intent['tag']:
                            print(chatbot.prompt, intent['responses'][states[tags.index(tag)]])
                            chatbot.user_context = intent['context_set']
                            if states[tags.index(tag)] < len(intent['responses'])-1:
                                states[tags.index(tag)] += 1
                            break
                    # Print response to intent with matching tag and no link to a context other than the current context (i.e., don't want to address intents outside of the current context) 
                    if tag == intent['tag'] and not 'context_filter' in intent: 
                        if tag == 'bot_name':
                            state = states[tags.index('bot_name')]
                            if state == 0:
                                print(chatbot.prompt, 'My name is ', chatbot.name, '! I will be your virtual assistant today here at GroundsKeeper Coffee, Co.', sep='')
                            elif state == 1:
                                print(chatbot.prompt, 'Sorry, I thought that I already told you! My name is ', chatbot.name, '! I will be your virtual assistant today here at GroundsKeeper Coffee, Co.', sep='')
                            else:
                                print(chatbot.prompt, 'I\'ve already told you, remember? I\'m ', chatbot.name, '!', sep='')
                            chatbot.user_context = intent['context_set']
                            states[tags.index('bot_name')] += 1
                        else:
                            print(chatbot.prompt, intent['responses'][states[tags.index(tag)]])
                            chatbot.user_context = intent['context_set']
                            if states[tags.index(tag)] < len(intent['responses'])-1:
                                states[tags.index(tag)] += 1
                    # Matching tag but context filter that does not match the current context
                    elif tag == intent['tag'] and chatbot.user_context != intent['context_filter']:
                        print(chatbot.prompt, 'Sorry, I\'m not understanding the context of what you\'re saying. Could you provide me with a bit more context, please?')


def main():
    chat()

main()

        


