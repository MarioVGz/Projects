import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Para suprimir mensajes de TensorFlow

import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import tensorflow as tf

lemmatizer = WordNetLemmatizer() # Crea una instancia de la class WordNetLemmatizer

# Configuración de TensorFlow para suprimir los mensajes de depuración
tf.get_logger().setLevel('ERROR')

# Importamos los archivos generados en el código anterior y los cargamos
intents = json.loads(open('intents.json', "r", encoding = "utf-8").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Pasamos las palabras de la oración a su forma raíz
# Se crea una funcion para limpiar el input del usuario
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence) 
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# Convertimos la información a unos y ceros según si están presentes en los patrones
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)  # Crea una lista de 0 con la longitud de la lista de palabras
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predecimos la categoría a la que pertenece la oración
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESGOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESGOLD] # Crea la lista de resultados
    results.sort(key=lambda x:x[1] , reverse = True) # Ordena la lista de los resultados por probabilidad del resultado
    return_list = [] 
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])}) # Agrega los intent y la probabilidad a la lista
    return return_list

# Obtenemos una respuesta aleatoria
def get_response(tag, intents_json):
    tag = tag[0]["intent"] # Obtiene la etiqueta de los intent
    list_of_intents = intents_json['intents'] # Obtiene lista de intents
    for i in list_of_intents:
        if i['tag'] == tag: # Si el tag es igual al tag del intent
            result = random.choice(i['responses']) # Obtiene una respuesta del intent
            break
    return result  # Retorna la respuesta

#def respuesta(message):
#    tag = predict_class(message)
#    res = get_response(tag, intents)
#    return res
