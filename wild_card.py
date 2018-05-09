import operator
from heapq import nlargest
import numpy
from keras import Sequential
from keras.layers import LSTM, Dense

def create_dataset(dataset, look_back=1):
    """
        Создает последовательность вида:
        X = [n-look_back, n-look_back+1, ...., n-1] и Y = [n]
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return numpy.array(dataX), numpy.array(dataY)


def extended_this(model, trainX, trainY, look_back, multi=2, type="extend"):
    """
        Продолжает последовательность в зависимости от type
        extend - продолжает
        remake - изменяет уже существующею
        continue - продолжает, удаляя оригинал
    """
    # Create dataset in comfortable type
    X = []
    Y = []
    new_Y = []
    for i in range(len(trainX)):
        X.append(trainX[i])
    for i in range(len(trainY)):
        Y.append(trainY[i])
    # Make len(X) * multi predictions
    new_l = len(X) * multi
    for i in range(int(len(X) * multi)):
        # Show every 100 iters
        if i % 100 == 0:
            print(i, "/", new_l)
        # Get last element in X. last
        last_x = numpy.array([X[len(X) - 1]])
        # Make prediction based on last_x
        last_y = model.predict_proba(last_x)
        # Create list which will contains new X = [n-look_back, n-look_back+1, ...., n-1]
        merge = []
        # Fill that new X
        for a in range(look_back - 1, 0, -1):
            merge.append(Y[i - a])
        # Add new prediction to new X
        merge.extend([last_y[0]])
        # Если мы хотим получить вектор с float
        # Y.append(last_y[0])
        # Если мы хотим получить вектор c 1 единицой и нулями
        top = nlargest(1, enumerate(last_y[0]), operator.itemgetter(1))
        top = top[0][0]
        print(top)
        last_y[0] = [0] * len(last_y[0])
        last_y[0][top] = 1
        new_Y.append(last_y[0])
        # [конец]Если мы хотим получить вектор c 1 единицой и нулями
        X.append(numpy.array(merge))
    # Return new datalist
    if type == "remake":
        return new_Y
    if type == "continue":
        v = len(Y) - len(new_Y)
        return new_Y[v:]
    if type == "extend":
        return Y
    return Y