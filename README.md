# Intelligent Systems (CS-5368-001) Project 3

## Table of Contents
[Description ](#description)

[Install Python](#install-python)

[Install NLTK](#install-nltk)

[Install PyTorch](#install-torch)

[Download Chatbot](#download-chatbot)

[Run Chatbot](#run-chatbot)

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
