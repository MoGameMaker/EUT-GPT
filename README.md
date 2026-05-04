<p align="center">
  <img src="https://img.shields.io/github/v/release/MoGameMaker/EUT-GPT-Test-1?include_prereleases&style=for-the-badge"/>
  <img src="https://img.shields.io/github/issues/MoGameMaker/EUT-GPT-Test-1?style=for-the-badge"/>
  <img src="https://img.shields.io/github/downloads/MoGameMaker/EUT-GPT-Test-1/total?style=for-the-badge"/>
</p>

# EUT-GPT Introduction

I (@mobogreatthegreat) have been working on EUT-GPT for the past few weeks now and I've finally decided to publish it publicly for testing and training by the EUT community.

# Table of Contents
- [Installation](#installation)
- [Installing from Source](#installing-from-source)
  - [Prerequisites](#prerequisites)
  - [Steps](#steps)
- [Installation Process](#installation-process)
- [Updating](#updating)
- [Getting Started](#getting-started)
  - [Chat Mode](#chat-mode)
  - [Train Mode](#train-mode)
- [Commands](#commands)
- [Updating the Global Train File (Next Update)](#updating-the-global-train-file-next-update)
- [Issues](#issues)
- [Transparency](#transparency)

# Installation
- On Windows: Head over to the Releases tab and select the version you want, then install the `.exe` file and run it.
- On Linux/MacOS: As I have not yet compiled a version for those 2 operating systems, you will need to download the source code and run it yourself to use EUT-GPT. Please refer to [Installing from Source](#installing-from-source) when using Linux/MacOS.

# Installing from Source
## Prerequisites
- Python 3.11+ (3.11 recommended)

## Steps
1. Download the source code from the Releases tab and extract the folder.
2. Open a terminal and navigate to the extracted folder.
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run `main.py` using this command:
```bash
python main.py
```

# Installation Process
After running the `.exe` file on Windows or running `main.py` on Linux/MacOS, the installation script will start.

During installation, the script will first install the Generation model (**DeepSeek R1 Distill Llama 8B**) from Hugging Face. This model is around *5 GB* in size.

After the model is done installing, the script will start to compile the wiki into a `.db` file. Both the model (saved as `model.gguf`) and the database file (saved as `WikiDump.db`) are stored in the Cache folder (`%APPDATA%\Local\EUTGPT` on Windows; Linux/MacOS location may vary).

If the script is interrupted, then you will need to go to the Cache folder and clear it (*Automatic Repair Mode Soon*).

# Getting Started
This section will be separated into 2 sections, one for Chat Mode and one for Train Mode.

After installation is complete, you will be prompted to select either **Chat mode** (by typing `Chat`) or **Train mode** (by typing `Train`).

If you are new, start with the [Chat Mode](#chat-mode) section.

# Updating
To update, all you need to do is download the new `.exe` file for Windows and it will work fine and for Linux/MacOS just follow the instructions in the [Installing from Source](#installing-from-source) section and transfer any new created files in the old directory to the new directory

## Chat Mode
**Chat Mode** is the mode you will usually use the most, serving as the interface between you and the Generation model.

When asking a question, you will be able to see the model's complete thinking and reasoning. The model's final response is usually preceded by `Final Answer:` or `Answer:`. It can still always be the final phrase given.

The interface has 2 things you can see:
- A prompt showing `You:` where you type your question
- `Assistant:` for the response

See the full list of available commands in the [Commands section](#commands).

## Train Mode
**Train Mode** is a mode made to enhance the model's accuracy by teaching it stuff the wiki doesn't have OR guiding it through the wiki with info it may not see as its search algorithm is not advanced/perfect.

The Train Mode interface has 3 inputs:
1. A prompt for the question to train on (you type the question here)
2. A prompt for the answer
3. An infinite entry prompt for wiki pages to check for this question (1 page name per prompt)

After you have given enough information, do `/endtrain` to end the training session.

Train files are stored in the same Cache folder as before as `train.jsonl`.

See also: [Commands](#commands)

# Commands
There are currently 6 available commands in EUT-GPT:

- `/updatewiki` — Recompiles the wiki database file using the newest version of the wiki  
- `/reloadmodel` — Reinstalls the Generation model  
- `/train` — Switch to train mode directly from chat mode  
- `/back` — Return to main menu interface  
- `/quit` — Closes the program  
- `/endtrain` — Ends training session in train mode  

*Useful references:*
- If something breaks, you may need to run `/updatewiki`
- If the model behaves incorrectly, try `/reloadmodel`
- If you want to switch modes quickly, use `/train` or `/back`

# Updating the Global Train File (Next Update)
If you would like to contribute your train file to be in the global train file, just make an issue with the enhancement tag and upload your `train.jsonl`.

If it's deemed good enough, it will be included in the next minor version (not patches/bug fixes) of EUT-GPT.

# Issues
If the model consistently gives incorrect or unexpected answers for the same question, then make an issue with the **bug** tag.

Please include:
- The exact question you asked  
- The incorrect or unexpected response  
- Any relevant training data that might be affecting the result (e.g. your `train.jsonl` file)

This helps identify whether the issue is caused by:
- The model itself  
- The wiki/database  
- Training data interference  

(deleting, modifying and disabling/enabling train files soon)

# Transparency
Some parts of EUT-GPT were coded using assistance from GPT-5.4 and Claude Haiku 4.5. Although there was use of AI during the development of this project, most of the code was written by me.

The main parts that AI was used in were for the System Prompt and the BM25 search algorithm. Part of this README was also written using the help of AI. The rest was written by me.
