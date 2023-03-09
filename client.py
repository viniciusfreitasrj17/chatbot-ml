# FOR PREDICTING RESPONSE #====================================================#

# For loading saved model
import nltk
import json
import pickle
import random
import warnings
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import numpy as np


nltk.download('punkt')  # Sentence tokenizer
lemmatizer = WordNetLemmatizer()
warnings.filterwarnings('ignore')
model = load_model('chatbot_model.h5')

# Preprocessing #==============================================================#

words = []
classes = []
documents = []
ignore_words = ['?', '!']
data_file = open('./intents.json').read()  # read json file
intents = json.loads(data_file)  # load json file


for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)# add each elements into list
        #combination between patterns and intents
        documents.append((w, intent['tag'])) # add single element into end of list
        # add to tag in our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

nltk.download('wordnet') #lexical database for the English language

nltk.download('omw-1.4')

# lemmatize, lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
# print (len(documents), "documents\n", documents, "\n")
# classes = intents[tag]
# print (len(classes), "classes\n", classes, "\n")
# words = all words, vocabulary
# print (len(words), "unique lemmatized words\n", words, "\n")
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))


#Utility Methods

def clean_up_sentence(sentence): # tokenize the pattern - split words into array

    sentence_words = nltk.word_tokenize(sentence)
    #print(sentence_words)
    # stem each word - create short form for word

    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    #print(sentence_words)

    return sentence_words
    #return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True): # tokenize the pattern

    sentence_words = clean_up_sentence(sentence)
    #print(sentence_words)

    # bag of words - matrix of N words, vocabulary matrix

    bag = [0]*len(words) 
    #print(bag)

    for s in sentence_words:  
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
                #print ("found in bag: %s" % w)
    #print(bag)
    return(np.array(bag))

def predict_class(sentence, model): # filter out predictions below a threshold

    p = bow(sentence, words,show_details=False)
    # print(p)

    res = model.predict(np.array([p]))[0]
    # print(res)

    ERROR_THRESHOLD = 0.25

    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # print(results)
    # sort by strength of probability

    results.sort(key=lambda x: x[1], reverse=True)
    # print(results)

    return_list = []

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    # print(return_list)
    return return_list

def getResponse(ints, intents_json):

    tag = ints[0]['intent']
    # print(tag)

    list_of_intents = intents_json['intents']
    # print(list_of_intents)

    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text): 
    ints = predict_class(text, model) 
    # print(ints)
    res = getResponse(ints, intents)
    # print(res)
    return res

start = True

while start:

    query = input('Enter Message:')
    if query in ['quit','exit','bye']:
        start = False
        continue
    try:
        res = chatbot_response(query)
        print(res)
    except:
        print('You may need to rephrase your question.')