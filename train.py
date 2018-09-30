import random
import ngram
import glob

from keras import Sequential
from keras.layers import LSTM, Dense
from music21 import converter, instrument, note, chord, stream
from music21.ext import joblib
import music21
from sklearn.preprocessing import LabelBinarizer

from heapq import nlargest
import operator
import wild_card
from music_controller import get_msg, create_midi

MELODY_LIMIT = 20

# Тут храним ноты
notes = []
# Счетчик для ограничения кол-ва мелодий
nn = 0
print('Engage!')
# Тусуемся в папке
for file in glob.glob("midi/mario/*.mid"):
    # Добавляем в хранилище еще ноты
    notes.extend(get_msg(file))
    nn+=1
    if nn > MELODY_LIMIT:
        break

print("Создаем словарь всех возможных нот...")
encoder = LabelBinarizer()
encoder.fit(notes)
joblib.dump(encoder, "encoders/LabelBinarizer.sav")
data = encoder.transform(notes)

print("Создаем ngram для поиска наиболее похожих нот...")
notes_set = set(notes)
ngram_notes = list(notes_set)
G = ngram.NGram(ngram_notes)
joblib.dump(G, "encoders/ngram.sav")

print("Создаем последовательности нот для обучения...")
look_back = 2
dataX, dataY = wild_card.create_dataset(data, look_back)

print("Перемешиваем мелодии...")
combined = list(zip(dataX, dataY))
random.shuffle(combined)
dataX[:], dataY[:] = zip(*combined)

print("Создаем тренировочные и тестовые данные...")
size_train = int(len(dataX)*0.8)
trainX = dataX[:size_train]
trainY = dataY[:size_train]

testX = dataX[size_train:]
testY = dataY[size_train:]

print('Создаем нейросеть...')
model = Sequential()
model.add(LSTM(64, input_shape=(look_back, len(data[0]))))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(data[0]), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

acc = []
print("Обучаем...")
for i in range(50):
    print("EPOCH " + str(i))
    out = model.fit(trainX, trainY, epochs=4, batch_size=32, verbose=1, shuffle=True)
    eval = model.evaluate(testX, testY, verbose=1)
    print(eval)
    acc.append(eval[1])
    model.save("models/mario" + str(eval[1]) + ".h5")

