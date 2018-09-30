import glob
import ngram
import numpy
from keras import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
from keras.utils import np_utils
from music21 import converter, instrument, note, chord, stream
from music21.ext import joblib
from sklearn.preprocessing import LabelBinarizer
import wild_card
from heapq import nlargest
import operator
from keras.models import load_model
from music_controller import create_midi, get_msg

notes = []
print("Engage!")
for file in glob.glob("input/*.mid"):
    notes.extend(get_msg(file))

create_midi(notes)

print("Загрузка ngram, для поиска похожих нот...")
G = joblib.load("encoders/ngram.sav")
for i in range(len(notes)):
    print(notes[i], G.find(notes[i]))
    notes[i] = G.find(notes[i])
create_midi(notes)

print("Берем из словаря коды для каждой ноты...")
encoder = LabelBinarizer()
encoder.fit(notes)
encoder = joblib.load("encoders/LabelBinarizer.sav")
data = encoder.transform(notes)

print("Создаем датасет...")
look_back = 2
trainX, trainY = wild_card.create_dataset(data, look_back)

print('Загружаем сеть...')
model = load_model("models/mario0.627370579342913.h5")

print("Генерируем...")
Y = wild_card.extended_this(model=model, trainX=trainX, trainY=trainY, look_back=look_back,
                              multi=1, type="remake")

print("Расшифруем полученые данные в мелодию...")
new_notes = []
text_labels = encoder.classes_
for i in range(len(Y)):
    # Ищем индекс самой вероятной ноты
    pred = Y[i]
    top = nlargest(1, enumerate(pred), operator.itemgetter(1))
    top = top[0][0]
    # Загружаем из словаря по индексу ноту
    predicted_label = text_labels[top]
    print(predicted_label)
    new_notes.append(predicted_label)

sequence_length = 100
create_midi(new_notes)
