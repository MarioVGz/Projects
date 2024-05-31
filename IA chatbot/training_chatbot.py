import random 
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

#Para crear la red neuronal
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers.schedules import ExponentialDecay
from keras.optimizers import SGD


intents = json.loads(open('intents.json', "r", encoding = "utf-8").read())

lemmatizer = WordNetLemmatizer()

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']

# Clasifica los patrones y las categorías
# Recorre cada intención y sus patrones en el archivo JSON
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokeniza las palabras en cada patrón y las agrega a la lista de palabras
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Agrega el par (patron, etiqueta) a la lista de documentos 
        documents.append((word_list, intent["tag"]))
        # Si la etiqueta no está en la lista de clases, la agrega
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lematiza las palabras y llas convierte en minúsculas, excluyendo las palabras ignoradas
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Guarda las listas de las palabras y clases en archivos pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Pasa la información a unos y ceros según las palabras presentes en cada categoría para hacer el entrenamiento
training = []
output_empty = [0]*len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Mezcla aleatoriamente el conjunto de entrenamiento
random.shuffle(training)

print(len(training)) 

# Divide el conjunto de entrenamiento en caracteristicas (train_x) y etiquetas (train_y)
train_x=[]
train_y=[]

for i in training:
    train_x.append(i[0])
    train_y.append(i[1])

train_x = np.array(train_x) 
train_y = np.array(train_y)

# Creamos la red neuronal
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), name="inp_layer", activation='relu'))
model.add(Dropout(0.5, name="hidden_layer1"))
model.add(Dense(64, name="hidden_layer2", activation='relu'))
model.add(Dropout(0.5, name="hidden_layer3"))
model.add(Dense(len(train_y[0]), name="output_layer", activation='softmax'))

lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)

#Creamos el optimizador y lo compilamos
sgd = SGD(learning_rate= lr_schedule, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

#Entrenamos el modelo y guardamos el modelo entrenado en un archivo h5
model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)
model.save("chatbot_model.h5")