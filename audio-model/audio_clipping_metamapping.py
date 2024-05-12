from pydub import AudioSegment
import os
from scipy import signal
import pandas as pd
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import librosa
from librosa import display
import pandas as pd
from sklearn.metrics import confusion_matrix
import keras
from keras.datasets import fashion_mnist
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Embedding
from keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D, LSTM, Bidirectional
from keras.optimizers import SGD
import tensorflow as tf
import numpy as np
import itertools
import math
import time
from Data_looping_patients import *

global number_list_train, phq8_array_train, count_train, number_list_test, phq8_array_test, count_test, samples_number

data = open(r'G:/metadata_mapped.csv', 'r', encoding='utf-8-sig')
mapping_data = pd.read_csv('G:/metadata_mapped.csv', header=0)
for number in samples_number:
    i = 1
    j = 0

    df = pd.read_csv('G:/data/' + str(number) + '_P/'
                     + str(number) + '_TRANSCRIPT.csv', header=0)
    while i < len(df) - 1:
        if j == 0:
            newAudio = AudioSegment.from_wav('G:/data/'
                                             + str(number) + '_P/' + str(number) + '_AUDIO.wav')
            t1 = float(df.iloc[0][0]) * 1000
            t2 = float(df.iloc[0][1]) * 1000
            newAudio = newAudio[t1:t2]
            newAudio.export('G:/data/Clipped/' + str(number) + '.wav', format="wav")
            j = 1
        else:

            newAudio = AudioSegment.from_wav('G:/data/'
                                             + str(number) + '_P/' + str(number) + '_AUDIO.wav')
            t1 = float(df.iloc[i][0]) * 1000
            t2 = float(df.iloc[i][1]) * 1000
            newAudio = newAudio[t1:t2]
            newAudio.export('G:/data/Clipped/' + str(number) + 'segmento.wav', format="wav")
            sound1 = AudioSegment.from_wav('G:/data/Clipped/' + str(number) + '.wav')
            sound2 = AudioSegment.from_wav('G:/data/Clipped/' + str(number) + 'segmento.wav')

            combined_sounds = sound1 + sound2
            combined_sounds.export('G:/data/Clipped/' + str(number) + '.wav', format="wav")
            i = i + 1
    os.remove('G:/data/Clipped/' + str(number) + 'segmento.wav')