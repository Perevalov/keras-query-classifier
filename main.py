from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation,Dropout,Embedding
import pandas as pd
import keras.utils
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.datasets import imdb

#объявляем константы
max_words = 1000
batch_size = 32
epochs = 3

#считываем из CSV
df = pd.read_csv('cleaned_dataset.csv',delimiter=';',encoding = "utf-8").astype(str)
num_classes = len(df['класс'].drop_duplicates())
X_raw = df['запрос'].values
Y_raw = df['класс'].values

#трансформируем текст запросов в матрицы
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_raw)
x_train = tokenizer.texts_to_matrix(X_raw)

#трансформируем классы
encoder = LabelEncoder()
encoder.fit(Y_raw)

encoded_Y = encoder.transform(Y_raw)
print(encoded_Y)
y_train = keras.utils.to_categorical(encoded_Y, num_classes)

#строим модель
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1)

model.save('classifier.h5')

