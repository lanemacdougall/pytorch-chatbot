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
    return model, all_words, tags

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
    model, all_words, tags = setup()
    chatbot = ChatBot(model, 'Kyle', all_words, tags)
    chatbot.greet()
    states = [0] * len(tags)
    while True:
        user_input = chatbot.get_input()
        if (user_input.lower() == 'done'):
            print(chatbot.prompt, 'Thanks for chatting! Have a great day!')
            break
        intent = chatbot.evaluate_input(user_input)
        if intent == 'fail':
            print(chatbot.prompt, 'Sorry, I didn\'t catch that. Can you say that again?')
        elif intent == 'greeting':
            state = states[tags.index('greeting')]
            if state == 0:
                print(chatbot.prompt, 'Hello! How can I be of assistance?')
            elif state == 1:
                print(chatbot.prompt, 'Hello again! I\'m still happy to be of assistance to you.')
            elif state == 2:
                print(chatbot.prompt, 'You really enjoy saying hello, huh? Well, so do I! Hello hello!')
            elif state == 3:
                print(chatbot.prompt, 'Still saying hello? Let\'s start getting to business! How can I help you?')
            else:
                print(chatbot.prompt, 'I guess quarantine really took a toll on you, huh?')
            states[tags.index('greeting')] += 1
        elif intent == 'goodbye':
            state = states[tags.index('goodbye')]
            if state == 0:
                print(chatbot.prompt, 'Bye, come back again soon.')
            elif state == 1:
                print(chatbot.prompt, 'I agree, goodbyes are hard, but feel free to come back any time!')
            elif state == 2:
                print(chatbot.prompt, 'Well I hope third time is the charm. See ya!')
            elif state == 3:
                print(chatbot.prompt, 'I should probably go talk to other customers, okay?')
            else:
                print(chatbot.prompt, 'Oh brother...')
            states[tags.index('goodbye')] += 1
        elif intent == 'thanks':
            state = states[tags.index('thanks')]
            if state == 0:
                print(chatbot.prompt, 'Happy to help!')
            elif state == 1:
                print(chatbot.prompt, 'You\'re so polite! I am here to serve.')
            else:
                print(chatbot.prompt, 'I think that I should be the one thanking you!')
            if state < 2:
                states[tags.index('thanks')] += 1
            else:
                states[tags.index('thanks')] = 0
        elif intent == 'how_are_you':
            state = states[tags.index('how_are_you')]
            if state == 0:
                print(chatbot.prompt, 'I am great, thank you!')
            elif state == 1:
                print(chatbot.prompt, 'Better now that I\'m talking to you!')
            elif state == 2:
                print(chatbot.prompt, 'Well, my battery is a little low, but otherwise I\'m doing good haha.')
            elif state == 3:
                print(chatbot.prompt, 'You are VERY concerned for my well-being! I\'m doing just fine.')
            else:
                print(chatbot.prompt, 'Okay enough about me, let\'s get down to business!')
            states[tags.index('how_are_you')] += 1
        elif intent == 'bot_name':
            state = states[tags.index('bot_name')]
            if state == 0:
                print(chatbot.prompt, 'My name is ', chatbot.name, '! I will be your virtual assistant today here at GroundsKeeper Coffee, Co.', sep='')
            elif state == 1:
                print(chatbot.prompt, 'Sorry, I thought that I already told you! My name is ', chatbot.name, '! I will be your virtual assistant today here at GroundsKeeper Coffee, Co.', sep='')
            else:
                print(chatbot.prompt, 'I\'ve already told you, remember? I\'m ', chatbot.name, '!', sep='')
            states[tags.index('bot_name')] += 1
        elif intent == 'options':
            state = states[tags.index('options')]
            if state == 0:
                print(chatbot.prompt, 'I am the virtual spokesperson for GroundsKeeper Coffee Supply, Co.!\n\tI am here to answer any of your questions concerning our business.')
            elif state == 1:
                print(chatbot.prompt, 'Well, as I said before, I\'m here to answer any of your questions concerning our business!')
            elif state == 2:
                print(chatbot.prompt, 'Like I said the other two times, I can answer any question you have about GroundsKeeper Coffee Supply, Co.')
            else:
                print(chatbot.prompt, 'Well, I can answer questions other than just this one! Haha. Kidding, of course. But I am curious about other questions that you might have.')
            states[tags.index('options')] += 1
        elif intent == 'coffee_supplies':
            state = states[tags.index('coffee_supplies')]
            if state == 0:
                print(chatbot.prompt, 'We offer large-scale coffee bean supplies in a variety of coffee bean types.\n\tCurrently, we have Ethiopian, Colombian, and Guatemalan.\n\tIf you would like to know more about these varieties, I would love to give you move information!')
            elif state == 1:
                print(chatbot.prompt, 'I so appreciate your curiousity! Like I said before, we offer large-scale coffee bean supplies in a variety of coffee bean types.\n\tRight now, we have Ethiopian, Colombian, and Guatemalan.\n\tIf you would like to know more about these varieties, I would love to give you move information!')
            elif state == 2:
                print(chatbot.prompt, 'You don\'t remember me telling you? We currently have Ethiopian, Colombian, and Guatemalan.\n\tAgain, if you would like to know more about these varieties, I would love to give you move information!')
            else:
                print(chatbot.prompt, 'We have Ethiopian, Colombian, and Guatamalan.\n\tAnd as I said before, if you would like to know more about these varieties, I would love to give you move information!')
            states[tags.index('coffee_supplies')] += 1
        elif intent == 'coffee_info':
            state = states[tags.index('coffee_info')]
            if state == 0:
                print(chatbot.prompt, 'Ethiopia is the birthplace of coffee. The coffee is known for its deep flavor, notes of chocolate, and floral aroma.\n\tColombian coffee is known for being medium-bodied and rich while also having a distinct citrus-like acidity.\n\tGuatemalan coffee is known for being full-bodied, strong, and full of flavor.')
            elif state == 1:
                print(chatbot.prompt, 'Well, as I said before, Ethiopia is the birthplace of coffee. The coffee is known for its deep flavor, notes of chocolate, and floral aroma.\n\tColombian coffee is known for being medium-bodied and rich while also having a distinct citrus-like acidity.\n\tGuatemalan coffee is known for being full-bodied, strong, and full of flavor.')
            else:
                print(chatbot.prompt, 'Again, Ethiopia is the birthplace of coffee. The coffee is known for its deep flavor, notes of chocolate, and floral aroma.\n\tColombian coffee is known for being medium-bodied and rich while also having a distinct citrus-like acidity.\n\tGuatemalan coffee is known for being full-bodied, strong, and full of flavor.')
            states[tags.index('coffee_info')] += 1
        elif intent == 'payment':
            state = states[tags.index('payment')]
            if state == 0:
                print(chatbot.prompt, 'We accept cash, personal checks, credit, PayPal, and CashApp.')
            elif state == 1:
                print(chatbot.prompt, 'Sorry, I thought I already told you! I apologize! We accept cash, personal checks, credit, PayPal, and CashApp.')
            elif state == 2:
                print(chatbot.prompt, 'I thought I already told you! We accept cash, personal checks, credit, PayPal, and CashApp.')
            else:
                print(chatbot.prompt, '*Exhale* Cash, personal checks, credit, PayPal, and CashApp.')
            states[tags.index('payment')] += 1
        elif intent == 'shipping_time':
            state = states[tags.index('shipping_time')]
            if state == 0:
                print(chatbot.prompt, 'Shipping will take approximately 2 - 6 days, depending on your location.')
            elif state == 1:
                print(chatbot.prompt, 'It depends on your location, but typically it takes 2 - 6 days.')
            elif state == 2:
                print(chatbot.prompt, 'Like I mentioned earlier, shipping will take approximately 2 - 6 days, depending on your location.')
            states[tags.index('shipping_time')] += 1
        elif intent == 'shipping_cost':
            state = states[tags.index('shipping_cost')]
            if state == 0:
                print(chatbot.prompt, 'Shipping cost depends entirely on the size of the shipment and your location.')
            elif state == 1:
                print(chatbot.prompt, 'It depends on your location and the size of your order, but I promise it\'s worth it!')
            elif state == 2:
                print(chatbot.prompt, 'Like I mentioned earlier, shipping cost depends entirely on the size of the shipment and your location.')
            states[tags.index('shipping_cost')] += 1
        elif intent == 'business_hours':
            state = states[tags.index('business_hours')]
            if state == 0:
                print(chatbot.prompt, 'We are open from 5 AM to 7 PM, Monday through Friday.')
            else:
                print(chatbot.prompt, 'Like I said earlier, we\'re open from 5 AM to 7 PM, Monday through Friday.')
            states[tags.index('business_hours')] += 1
        elif intent == 'rep_times':
            state = states[tags.index('rep_times')]
            if state == 0:
                print(chatbot.prompt, 'You may speak with a representative from 5 AM to 7 PM, Monday through Saturday at 1-800-GROUNDS.')
            else:
                print(chatbot.prompt, 'Like I said earlier, you may speak with a representative from 5 AM to 7 PM, Monday through Saturday at 1-800-GROUNDS.')
            states[tags.index('rep_times')] += 1
        elif intent == 'representative':
            state = states[tags.index('representative')]
            if state == 0:
                print(chatbot.prompt, 'Excellent! Our representatives are the best in the business. You may speak with a representative from 5 AM to 7 PM, Monday through Saturday at 1-800-GROUNDS.')
            else:
                print(chatbot.prompt, 'I\'m so glad that you would like to speak with a representative. You may speak with a representative from 5 AM to 7 PM, Monday through Saturday at 1-800-GROUNDS.')
            states[tags.index('representative')] += 1
        elif intent == 'no_answer':
            state = states[tags.index('no_answer')]
            if state == 0:
                print(chatbot.prompt, 'Sorry, did you mean to say something?')
            elif state == 1:
                print(chatbot.prompt, 'Just up your questions and I\'ll be happy to answer them!')
            else:
                print(chatbot.prompt, 'Hmmm, I\'m just seeing that you\'re entering blanks... Is there an issue on your end? All seems to be good on this end!')
            states[tags.index('no_answer')] += 1
        #TODO: Get rid of the below 
        else:
            print(intent)



def main():
    chat()

main()

        


