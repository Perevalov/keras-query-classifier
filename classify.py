import sys,os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from keras.models import load_model
import numpy as np
from keras.preprocessing.text import Tokenizer
import pandas as pd
import argparse,stop_words_remover as swr


classes = {'Автоистория': 0,'Автострахование': 1, 'ВУ': 2, 'Жалобы':3, 'Запись в ГИБДД':4, 'Запись в МАДИ':5,'Запись на медкомиссию':6
            ,'Нарушения и штрафы':7,'Обращения в МАДИ и АМПП':8,'ПТС':9,'Регистрация':10,'Статус регистрации':11,
           'Такси':12,'Эвакуация':13}

model = load_model('classifier.h5')

def predict(str_query,numwords):

    tokenizer = Tokenizer(num_words=numwords)
    X_raw_test = [str_query]
    df = pd.read_csv('cleaned_dataset.csv', delimiter=';', encoding="utf-8").astype(str)
    X_raw = df['запрос'].values
    tokenizer.fit_on_texts(X_raw)
    x_test = tokenizer.texts_to_matrix(X_raw_test, mode='binary')
    prediction = model.predict(np.array(x_test))
    class_num = np.argmax(prediction[0])
    sys.stderr = stderr
    for name, index in classes.items():
        if index == class_num:
            print(name)

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('query', type=str, help="a query to classify")
args =parser.parse_args()

query = swr.remove_stop_words(args.query)

predict(query,1000)