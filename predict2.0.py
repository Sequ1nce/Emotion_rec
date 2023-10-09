import os
import pandas as pd
import numpy as np
import matplotlib as plt

#%%

import librosa
import librosa.display
import time
from tqdm import tqdm

#%%

from sklearn.utils import shuffle
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

#%%

import tensorflow as tf
from matplotlib.pyplot import specgram
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import confusion_matrix

from keras.layers import LSTM as KERAS_LSTM
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import opensmile

from pyaudio1 import rec

import threading
from threading import Thread
import time

def calc_features(X, sample_rate):
    # mfcc features. mean as the feature. Could do min and max etc as well.
    mfccs = np.mean(librosa.feature.mfcc(y=X,
                                         sr=sample_rate,
                                         n_mfcc=13),
                    axis=0)

    if mfccs.shape[0] < 216:
        ln = mfccs.shape[0]
        mfccs = np.pad(mfccs, (0, 216 - len(mfccs)), mode='empty')
        for i in range(ln + 1, 216):
            mfccs[i] = np.nan

    # opensmile features
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    smiles = smile.process_signal(
        X,
        sample_rate)
    npsmiles = smiles.to_numpy(dtype=float)
    npsmiles = npsmiles[0]
    if npsmiles.shape[0] < 88:
        ln = npsmiles.shape[0]
        npsmiles = np.pad(npsmiles, (0, 88 - len(npsmiles)), mode='empty')
        for i in range(ln + 1, 88):
            npsmiles[i] = np.nan

    feats = np.concatenate(([mfccs], [npsmiles]), axis=1)

    data_df = pd.DataFrame(feats)

    return data_df

def predict(data_df, norm_coeffs_path):

    rnewdf = data_df

    rnewdf.drop(rnewdf.iloc[:, 78:216], inplace=True, axis=1)  # 78-216ceps
    cols = rnewdf.columns
    rnewdf.drop(columns=[292], inplace=True, axis=1)
    rnewdf = rnewdf.dropna(axis=0)  #

    norm_coef_df = pd.read_csv(norm_coeffs_path)
    norm_coef_df = norm_coef_df.drop(columns=['Unnamed: 0'])
    cols = norm_coef_df.columns
    # print(rnewdf[c])
    for i in range(0, 30, 3):
        med = norm_coef_df[cols[i]].to_numpy(dtype=float)
        perDown = norm_coef_df[cols[i + 1]].to_numpy(dtype=float)
        perUp = norm_coef_df[cols[i + 2]].to_numpy(dtype=float)

        rnewdf = (rnewdf - med) / (perUp - perDown)

    arr = np.array([[10000], [1]]).transpose().astype('float32')
    df = pd.DataFrame(arr, columns=['dictor', 'gender'])

    rnewdf = pd.concat([df, rnewdf], axis=1)

    X_test = np.array(rnewdf)
    x_testcnn = np.expand_dims(X_test, axis=2)

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # model = Sequential()

    model = Sequential()
    model.add(Conv1D(256, 5, padding='same',
                     input_shape=(167, 1)))
    model.add(Activation('LeakyReLU'))
    model.add(Conv1D(128, 5, padding='same'))
    model.add(Activation('LeakyReLU'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(Conv1D(128, 5, padding='same', ))
    model.add(Activation('LeakyReLU'))
    model.add(Conv1D(128, 5, padding='same', ))
    model.add(Activation('LeakyReLU'))
    model.add(Flatten())
    model.add(Dense(6))
    model.add(Activation('softmax'))
    opt = keras.optimizers.RMSprop(learning_rate=0.00001, decay=1e-6)
    model.load_weights('rav_crema_cnn03.h5')
    #model.summary()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    preds = model.predict(x_testcnn,
                          batch_size=16)

    print('PREDS = ' + np.array2string(preds, suppress_small=True))

    y_pred = np.argmax(preds, axis=1)

    return str(y_pred)

def predict_emotion(X, sample_rate, norm_coeffs_path):
    data_df = calc_features(X, sample_rate)
    y_pred = predict(data_df, norm_coeffs_path)
    return y_pred

def pred_rec(n):
    while n > 0:
        path = rec(n)
        print('predict')

        n += 2
        X, sample_rate = librosa.load(path
                                      , res_type='kaiser_fast'
                                      , duration=2.5
                                      , sr=44100
                                      , offset=0.5
                                      )

        sample_rate = np.array(sample_rate)
        norm_coeffs_path = 'norm_coeffs_TESS_RAV.csv'
        y_pred = predict_emotion(X, sample_rate, norm_coeffs_path)
        time.sleep(1.7)

def get_txt(filename):
    try:
        f = open(filename, encoding='utf - 8')
        lis = f.readlines()
        result = []
        for line in lis:
            if 'PREDS' in line:
                spline = line.split(' = ')
                arr = spline[1].replace('[', '').replace(']', '').replace(']\n', '')
                result.append(np.fromstring(arr, sep=' '))
    finally:
        f.close()
        return result

def rand_forest():
    data = get_txt('C:\\Users\\Public\\Documents\\pythonProject2\\angry\\angrytxt\\doc6.txt')
    df = pd.DataFrame(data, columns=['angry', 'disg', 'fear', 'happy', 'neutral', 'sad'])

    X_train, X_test, y_train, y_test = train_test_split(data, Y, test_size=0.2)
    rf = RandomForestClassifier(n_jobs=-1)
    parameters = {'n_estimators': range(20, 30, 2), 'max_depth': range(5, 10), 'min_samples_leaf': range(1, 7),'min_samples_split': [2, 4]}
    gscv_rf = GridSearchCV(rf, param_grid=parameters, cv=3, n_jobs=-1)
    gscv_rf.fit(X_train, np.ravel(y_train))
    best_rf = gscv_rf.best_estimator_
    result=best_rf.predict(X_test)

    return result

if __name__ == '__main__':

    first = Thread(target=pred_rec, args=(1,))
    # first.daemon = True
    first.start()
    time.sleep(2.5)
    second = Thread(target=pred_rec, args=(2,))
    # second.daemon = True
    second.start()

    # ang = 0
    # disg = 1
    # fear = 2
    # happy = 3
    # ney = 4
    # sad = 5

    # if __name__ == '__main__':
    #     n = 1
    #     while n > 0:
    #
    #         path = rec(n)
    #
    #
    #
    #         #ang = 0
    #         #disg = 1
    #         #fear = 2
    #         #happy = 3
    #         #ney = 4
    #         #sad = 5
    #
    #         print('predict')
    #         # path = 'C:\\Users\\Public\\Documents\\pythonProject2\\angry\\output' + str(n) + '.wav'
    #         n += 1
    #         X, sample_rate = librosa.load(path
    #                                       , res_type='kaiser_fast'
    #                                       , duration=2.5
    #                                       , sr=44100
    #                                       , offset=0.5
    #                                       )
    #
    #         sample_rate = np.array(sample_rate)
    #         norm_coeffs_path = 'norm_coeffs_TESS_RAV.csv'
    #         y_pred = predict_emotion(X, sample_rate, norm_coeffs_path)
    # %%
