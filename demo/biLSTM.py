from collections import Counter
import numpy as np
from csv import DictReader
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import Sequential, Model, load_model
from keras.layers import concatenate, Embedding, Dense, Dropout, Activation, LSTM, Flatten, Input, RepeatVector, TimeDistributed, Bidirectional
from keras.optimizers import Adam, RMSprop
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard
import codecs
import pickle

MAX_LEN_HEAD = 100
MAX_LEN_BODY = 500
VOCAB_SIZE = 15000
EMBEDDING_DIM = 300
label_ref = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}

def get_vocab(lst, vocab_size):
    """
    lst: list of sentences
    """
    vocabcount = Counter(w for txt in lst for w in txt.lower().split())
    vocabcount = vocabcount.most_common(vocab_size)
    word2idx = {}
    idx2word = {}
    for i, word in enumerate(vocabcount):
        word2idx[word[0]] = i
        idx2word[i] = word[0]
    return word2idx, idx2word

def cov2idx_common(lst, word2idx, max_len):
    output = []
    for sentence in lst:
        temp = []
        counter = Counter(sentence.split())
        for wordPair in counter.most_common(max_len):
            if wordPair[0] in word2idx:
                temp.append(word2idx[wordPair[0]])
        output.append(temp)
    return output

def pad_seq(cov_lst, max_len=MAX_LEN_BODY):
    """
    list of list of index converted from words
    """
    pad_lst = pad_sequences(cov_lst, maxlen = max_len, padding='post')
    return pad_lst

def load_test_data(list_headlines, body, word2idx):
    cov_head = cov2idx_common(list_headlines, word2idx, MAX_LEN_HEAD)
    cov_body = cov2idx_common([body], word2idx, MAX_LEN_BODY)
    pad_head = pad_seq(cov_head, MAX_LEN_HEAD)
    pad_body = pad_seq(cov_body, MAX_LEN_BODY)
    pad_body = pad_body * len(pad_head)
    return pad_head, pad_body

class Model:
    def load_model(self, model_location):
        self.model = load_model(model_location)

    def predict(self, list_headlines, body, word2idx=None):
        if not word2idx:
            word2idx = pickle.load(open("word2idx.pkl", "rb"))
        pad_head, pad_body = load_test_data(list_headlines, body, word2idx)
        pred = self.model.predict([pad_head, pad_body])
        return np.argmax(pred, axis = 1)
