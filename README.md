# Intelligent Systems (CS-5368-001) Project 3

## Table of Contents
[Description ](#description)

[Install Python](#install-python)

[Install NLTK](#install-nltk)

[Install PyTorch](#install-torch)

[Download Chatbot](#download-chatbot)

[Run Chatbot](#run-chatbot)

[Tips When Using Chatbot](#tips)

## <a name="description"></a> Description
Basic retrieval-based conversational agent (chatbot) intended to be used by a fictional bulk coffee supplier, GroundsKeeper Coffee, Co.
The chatbot, nicknamed Bean, is capable of recording users' orders and answering questions about the business' available coffee varieties and hours of operation, the availability of customer service representatives, as well as shipping and payment.

This retrieval-based chatbot is implemented using PyTorch and Natural Language Toolkit (NLTK). Natural language processing (NLP) techniques performed on training data and user input is done using Python NLTK. A neural network model, implemented using the PyTorch framework and trained on the data in [intents.json](https://github.com/lanemacdougall/project-3-chatbot/blob/main/intents.json), enables the chatbot to perform intent classification on user input. The predicted intent is then used to retrieve an appropriate response. 

## <a name="install-python"></a> Install Python
1. Go to [python.org/downloads](https://www.python.org/downloads/).
2. Download the latest version of Python 3 (Python 3.9.0 at the time of this writing) for your specific operating system (e.g., Mac OS X, Windows, Linux).
3. Use this [link](https://realpython.com/installing-python/) to guide you through the process of installing and setting-up Python 3 on your specific OS.

## <a name="install-nltk"></a> Install NLTK
1. Ensure that Python has been successfully installed on your computer (see how using this [link](https://realpython.com/installing-python/)).
2. Ensure that NumPy is installed on your computer (if you followed the above instructions on installing Python 3, NumPy will be installed on your computer). If you are unsure if NumPy is installed on your computer, follow this [link](https://stackoverflow.com/questions/5016186/how-to-detect-if-numpy-is-installed) for information on how you can check if it is.
3. Go to [NLTK's website](https://www.nltk.org/install.html) and follow the instructions corresponding to your specific OS.

## <a name="install-torch"></a> Install PyTorch
1. Go to [PyTorch's website](https://pytorch.org/get-started/locally/), select your desired specifications under the "Start Locally" section, and then follow the installation instructions corresponding to your specific OS.

## <a name="download-project"></a> Download Chatbot
Two download options are available to you.
### Clone Repository
1. Open the terminal or command prompt. [More information](https://www.groovypost.com/howto/open-command-window-terminal-window-specific-folder-windows-mac-linux/).
2. Navigate to the directory where you would like to store the cloned repository.
3. Enter the following command into the terminal or command prompt:
```sh
$ git clone https://github.com/lanemacdougall/project-3-chatbot
```

### Download Project
1. Go to https://github.com/lanemacdougall/project-3-chatbot.
2. Above the list of files, click &#8595; Code.
3. Click "Download ZIP".
4. Save the ZIP file in the directory where you would like to store the project.
5. Extract the ZIP file.

## <a name="run-chatbot"></a> Run Chatbot
1. Open the terminal or command prompt. [More information](https://www.groovypost.com/howto/open-command-window-terminal-window-specific-folder-windows-mac-linux/).
2. Navigate to the directory where the program is stored.
3. Enter the following command into the terminal or command prompt:
```sh
$ python3 chat_bot.py
```

## <a name="tips"></a> Tips When Using Chatbot
1. Try out the latest feature added to the chatbot: You can add to an order that you've already placed with the chatbot (within a single conversation/program execution). Meaning, if you place an order for 2 pounds of Colombian coffee with the chatbot and then carry on with other conversation, you may later request to add to your order and then add 3 pounds of Ethiopian coffee and an additional 2 pounds of Colombian coffee to your order. You can of course add any number of the offered varieties in any valid weights.
2. You are able to cancel your order after you tell the chatbot that you would like to order or after you specify the variety(ies) that you would like to order. However, once an order is placed with the chatbot, it cannot be cancelled.
3. The chatbot is programmed to be able to respond to questions and comments regarding the coffee shop that it is intended to be used by, as well as general greetings and formalities. Suggested inputs that you may want to try out include: "Hello, Bean!", "Hey there, Bean!", "How are you doing today?", "Are you doing good today, Bean?", "Thanks so much for all your help today!", "How are you able to assist me?", "What varieties of coffee do you currently have in inventory?", "I have decided what I would like to order today!", "When are your hours of operation?", "What forms of payment do you accept?", "How long does shipping take?", "How much does shipping cost?", "When can I speak to a customer service representative?", and "I want to speak to a rep right now!". 
4. Do not overwhelm the chatbot with multiple questions and/or comments that might be linked to different intents in a single input. The chatbot is not designed to respond to multiple intents at one time.
5. Feel free to make mistakes. The chatbot is resilient to mistakes, and you can backtrack or change topics of conversation at any time.
