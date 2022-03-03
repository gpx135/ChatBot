import nltk 
nltk.download('punkt')

from nltk import word_tokenize, sent_tokenize

from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np 
import tflearn
import random
import json
import pickle
from googletrans import Translator

translator = Translator()


with open("questions.json") as file:
    data = json.load(file)

with open("data.pickle","rb") as f:
    words, labels, training, output = pickle.load(f)

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.load("model.tflearn")



def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    
    return np.array(bag)


def chat():
    print("Start talking with the bot! (type quit to stop)")
    while True:
        inp = input("You: ")
        ori_lang = translator.detect(inp).lang
        inp_translated = translator.translate(inp, dest='en').text

        if inp.lower() == "quit":
            break

        result = model.predict([bag_of_words(inp_translated, words)])[0]
        result_index = np.argmax(result)
        tag = labels[result_index]

        if result[result_index] > 0.7:
            for tg in data["questions"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            print(translator.translate(random.choice(responses), src='en', dest=ori_lang).text)

        else:
            print(translator.translate("Sorry, I don't understand. Can you please rephrase it?", src='en', dest=ori_lang).text)


chat()