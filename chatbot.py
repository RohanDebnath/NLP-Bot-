# chatbot.py
import tkinter as tk
from tkinter import Text, Entry, Button, END
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

class Chatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.words = pickle.load(open('words.pkl', 'rb'))
        self.classes = pickle.load(open('classes.pkl', 'rb'))
        self.model = load_model('chatbot_model.h5')
        
        try:
            with open('intents.json', 'r') as file:
                self.intents = json.load(file)
        except FileNotFoundError:
            print("The file 'intents.json' was not found.")
            self.intents = None
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            self.intents = None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            self.intents = None

    def clean_up_sentences(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence):
        sentence_words = self.clean_up_sentences(sentence)
        bag = [0] * len(self.words)
        for w in sentence_words:
            for i, word in enumerate(self.words):
                if word == w:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        result = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

        result.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in result:
            return_list.append({'intent': self.classes[r[0]], 'probability': str(r[1])})
        return return_list

    def get_response(self, intent_list):
        tag = intent_list[0]['intent']
        list_of_intents = self.intents['intents']
        for intent in list_of_intents:
            if intent['tags'] == tag:
                result = random.choice(intent['responses'])
                break
        return result

class ChatGUI:
    def __init__(self, master, chatbot):
        self.master = master
        self.master.title("Chatbot GUI")
        self.chatbot = chatbot

        self.chat_log = Text(master, state='disabled', wrap='word', font=("Arial", 12))
        self.chat_log.pack(expand=True, fill=tk.BOTH)

        self.user_input = Entry(master, font=("Arial", 12))
        self.user_input.pack(expand=True, fill=tk.X, side=tk.LEFT)
        self.user_input.bind("<Return>", lambda event: self.send_message())

        self.send_button = Button(master, text="Send", command=self.send_message)
        self.send_button.pack(side=tk.RIGHT)

    def send_message(self):
        user_message = self.user_input.get()
        self.user_input.delete(0, END)

        self.update_chat_log(f"You: {user_message}")

        # Get the chatbot's response
        intent_list = self.chatbot.predict_class(user_message)
        bot_response = self.chatbot.get_response(intent_list)

        self.update_chat_log(f"Chatbot: {bot_response}")

    def update_chat_log(self, message):
        self.chat_log.config(state='normal')
        self.chat_log.insert(END, message + '\n')
        self.chat_log.config(state='disabled')
        self.chat_log.see(END)

if __name__ == "__main__":
    root = tk.Tk()
    chatbot_instance = Chatbot()
    chat_gui = ChatGUI(root, chatbot_instance)
    root.mainloop()
