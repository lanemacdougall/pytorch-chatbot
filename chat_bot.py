import random
import json
import torch
from chatbot_model import NeuralNetwork
from nltk_utility import tokenize, bag_of_words

# Chatbot implementation broken up into Chatbot class and setup and chat methods

# setup method sets up environment by setting device and loading database of responses and model
def setup():
    
    # GPU support is not available on the device used to develop this program - switch to if-else statement if 'cuda' support may be available
    device = torch.device('cpu')

    # intents.json acts as a database of responses in this file - contains responses for each intent
    with open('intents.json', 'r') as file:
        intents = json.load(file)

    # Load model data saved after training
    FILE = "model.pth"
    model_data = torch.load(FILE)

    input_size = model_data['input_size']
    hidden_size1 = model_data['hidden_size1']
    hidden_size2 = model_data['hidden_size2']
    output_size = model_data['output_size']

    all_words = model_data['all_words']
    tags = model_data['tags']
    model_state = model_data['model_state']

    # Instantiate model and load model state saved after training
    model = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size).to(device)
    model.load_state_dict(model_state)
    # Set model to evaluation mode - parameters are not updated
    model.eval()
    return model, all_words, tags, intents  # all_words, tags, and intents needed by chat method


# Encapsulate data and operations required by chatbot functionality into a class
class ChatBot():
    def __init__(self, model, name, all_words, tags):
        self.model = model
        self.all_words = all_words
        self.tags = tags
        self.user_context = ''  # Updated according to the context of the conversation between the user and the chatbot - provides context to user input that lacks context
        # The below are included as ChatBot object attributes to allow for changes to be made easily
        self.name = name
        self.prompt = 'Bot: '
        self.greeting = 'Welcome to GroundsKeeper Coffee, Co.! My name is ' + self.name + '.'
        self.abilities = 'I am here to answer questions concerning our available coffee varieties, our hours of operation, the availability of our customer service representatives, shipping, and payment.\n\
Also, whenever you\'re ready, I am here to take your order.\nIf you ever forget these options and want a reminder, just let me know!'
        
    # Method prints initial greeting and instructions at the start of the conversation
    def greet(self):
        print("\n" + self.greeting, end="\n\n")
        print(self.abilities, end="\n\n")
        print('Enter \'done\' to exit.\n')

    # Method gets input from the user
    def get_input(self):
        user_in = input('You: ')
        return user_in

    # Method processes the user input, passes the processed input into the model, and returns the predicted intent
    def evaluate_input(self, user_in):
        # Tokenize user input and generate a bag of words for the input (relative to the words in the data set that the model was trained on)
        tok_input = tokenize(user_in)
        bag = bag_of_words(tok_input, self.all_words)
        
        bag = bag.reshape(1, bag.shape[0])  #Reshape numpy array to match shape of neural network's input
        bag = torch.from_numpy(bag)  # Convert numpy array to tensor
        output = self.model(bag)  # Pass tensor into model and retrieve output
        _, predicted = torch.max(output, dim=1)  # Apply torch.max method to retrieve the largest value in the output tensor and its index (we only store the index)

        intent = self.tags[predicted.item()]  # Retrieve the intent tag corresponding the index returned by the torch.max method (i.e., the predicted intent tag)
        
        probs = torch.softmax(output, dim=1)  # Convert the model output into probabilities using the torch.softmax method
        intent_prob = probs[0][predicted.item()]  # Retrieve the probability corresponding to the predicted intent tag from the probabilities returned by the torch.softmax method

        # If the model's prediction has a probability greater than 75%, return the predicted intent tag; otherwise return 'fail'
        if intent_prob > 0.75:
            return intent
        else:
            return 'fail'

# chat method drives the Chatbot object's conversation with the user
def chat():
    model, all_words, tags, intents = setup()  # Call the setup method and retrieve the model, the all_words and tags arrays, and the intents JSON object that it returns
    chatbot = ChatBot(model, 'Bean', all_words, tags)  # Instantiate the Chatbot object with the model, some arbitrary name, and the tags array
    chatbot.greet()  # Print initial greeting and instructions at the start of the conversation
    states = [0] * len(tags)  # Array storing n integers (where n is the number of intent tags) that represent the number of times user input has corresponded to a given intent
                              # The chatbot's response for an intent is determined by the corresponding number in the states array; i.e., the response changes as the user repeats a given question or statement
                              # This trick gives the illusion that the chatbot remembers the questions that it is asked and understands that it has already offered a response for repeated questions/comments
    
    # Arrays storing the varieties and weights of the user's coffee order
    selected_varieties = []
    selected_weights = []
    while True:
        user_input = chatbot.get_input()
        # 'done' signals to the system that the user wishes to discontinue program execution
        if user_input.lower() == 'done':
            print(chatbot.prompt, 'Thanks for chatting! Have a great day!')
            break
        # Evaluate (process, pass into model, and retrieve model's prediction) user input; tag is the predicted intent tag
        tag = chatbot.evaluate_input(user_input)
        
        # If the context of the conversation has been set to select_variety, the system is going to retrieve the coffee variety(ies) that the user wishes to order
        if chatbot.user_context == 'select_variety':
            # User can cancel order (backtrack) using natural language (i.e., 'No thank you', 'Nevermind', 'I don't want any', 'I have changed my mind', etc.)
            if 'no' in user_input.lower() or 'not' in user_input.lower() or 'don\'t' in user_input.lower() or 'nah' in user_input.lower() or 'pass' in user_input.lower() or 'nevermind' in user_input.lower() or ('changed' in user_input.lower() and 'mind' in user_input.lower()):
                print(chatbot.prompt, 'Okay! How else can I assist you?')
                chatbot.user_context = ''  # Context is reset and user is prompted to provide new input
                continue
            # Determine which variety(ies) of coffee the user has specified and store it (them) in the selected_varieties array
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
            # If none of the available varieties are specified by the user, display message saying so and re-prompt the user for their desired variety(ies)
            else:
                print(chatbot.prompt, 'I\'m sorry we only offer Ethiopian, Colombian, and Guatemalan varieties at this time. Please try again.')
            
            # Prompt user to enter weights corresponding to each of their selected varieties
            if len(selected_varieties) == 1:
                print(chatbot.prompt, 'Perfect. And what will be the weight (in pounds)?')
                chatbot.user_context = 'select_weight'  # Set context to select_weight
            elif len(selected_varieties) > 1:
                print(chatbot.prompt, 'Perfect. Please enter the weights (in pounds) for the ', end="")
                for i in selected_varieties:
                    print(i, end=", ")
                print("in that order and separated by a comma and a space.")
                chatbot.user_context = 'select_weight'  # Set context to select_weight

        # If the context of the conversation has been set to select_weight, the system is going to retrieve the weight(s) corresponding to the coffee variety(ies) that the user wishes to order
        elif chatbot.user_context == 'select_weight':
            # User can cancel order (backtrack) using natural language (i.e., 'No thank you', 'Nevermind', 'I don't want any', 'I have changed my mind', etc.)
            if 'no' in user_input.lower() or 'not' in user_input.lower() or 'don\'t' in user_input.lower() or 'nah' in user_input.lower() or 'pass' in user_input.lower() or 'nevermind' in user_input.lower() or ('changed' in user_input.lower() and 'mind' in user_input.lower()):
                selected_varieties = []  # Clear the selected_varieties array - essentially clear the order
                print(chatbot.prompt, 'Okay! How else can I assist you?')
                chatbot.user_context = ''  # Context is reset and user is prompted to provide new input
            else:
                # Extract integers and floating-point numbers (representing weights in pounds) from the user input
                for word in user_input.lower().split():
                    num = word.replace(",", "", 1)
                    altered_num = num.replace(".", "", 1)
                    if altered_num.isdigit():
                        selected_weights.append(num)
                # Provide error message and re-prompt user if no weights (valid numbers) can be extracted from user input or if the number of weights (valid numbers) entered by the user 
                # do not equal the number of varieties selected by the user
                if len(selected_weights) == 0:
                    print(chatbot.prompt, 'I\'m sorry, I\'m not seeing any valid weights. Please try again. Remember to list multiple weights separated by a comma and a space.')
                elif len(selected_weights) != len(selected_varieties):
                    print(chatbot.prompt, 'Please enter a weight for each of the coffee varieties that you selected for your order. Remember to list multiple weights separated by a comma and a space.')
                    selected_weights = []
                else:
                    # Zip the selected_varieties and selected_weights arrays together and convert to dictionary - this dictionary represents the user's order
                    order = dict(zip(selected_varieties, selected_weights))
                    # Print out user's order - if-else blocks are used for proper formatting of printed strings
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
                    chatbot.user_context = ''  # Context is reset and user is prompted to provide new input

        # Processing carried out if user_context is not equal to 'select_variety' or 'select_weight'
        else:
            # If predicted intent tag did not have a probability greater than the specified threshold, print error message
            if tag == 'fail' or user_input == '':
                print(chatbot.prompt, 'Sorry, I didn\'t catch that. Your question or comment may be outside of my scope of understanding.\nIf you had a typing error or are able to re-phrase your question or comment, please try again. Otherwise, please feel free to contact my developer with ways in which you believe I can be improved!')
            else:
                # Iterate over intents in intents JSON object and find the one corresponding to the predicted intent tag and current context of conversation
                for intent in intents['intents']:
                    # Priority given to intents linked to current context
                    if 'context_filter' in intent and chatbot.user_context == intent['context_filter']:
                        if tag == intent['tag']:
                            print(chatbot.prompt, intent['responses'][states[tags.index(tag)]])  # Print the response corresponding to the state of the intent (see comments on lines 92 thru 94 for more details)
                            chatbot.user_context = intent['context_set']  # Set context specified by the intent
                            # If not on the last response corresponding to the intent, increment the value of the intent's state so that a new response will be generated the next time the user triggers this intent
                            if states[tags.index(tag)] < len(intent['responses'])-1: 
                                states[tags.index(tag)] += 1
                            break  # Halt processing 
                    # Print response to intent with matching tag and no link to a context (i.e., don't want to address intents outside of the current context) 
                    if tag == intent['tag'] and not 'context_filter' in intent: 
                        # Responses for 'bot_name' intent tag are outside of the intents JSON object (and file) because of the variables that must be included in the response (i.e., filling in a template)
                        # This section is useful in better understanding how the state array is used in determining the responses generated by the chatbot and how those responses change as the user repeats
                        # questions and comments
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
                            print(chatbot.prompt, intent['responses'][states[tags.index(tag)]])  # Retrieve and print the response corresponding to the intent's state from the intents JSON object
                            chatbot.user_context = intent['context_set']  # Set context specified by the intent
                            # If not on the last response corresponding to the intent, increment the value of the intent's state so that a new response will be generated the next time the user triggers this intent
                            if states[tags.index(tag)] < len(intent['responses'])-1:
                                states[tags.index(tag)] += 1

                    # Matching tag but context filter that does not match the current context - print message asking user for more context
                    elif tag == intent['tag'] and chatbot.user_context != intent['context_filter']:
                        print(chatbot.prompt, 'Sorry, I\'m not understanding the context of what you\'re saying. Could you provide me with a bit more context, please?')


# Begin conversation on program run
def main():
    chat()

main()

        


