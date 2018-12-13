from collections import Counter
import numpy as np
from csv import DictReader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.models import Sequential, Model, load_model
from keras.layers import concatenate, Embedding, Dense, Dropout, Lambda, Activation, LSTM, Flatten, Input, RepeatVector, TimeDistributed, Bidirectional
from keras.optimizers import Adam, RMSprop
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard
import keras.backend as K
import codecs
import pickle



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
        self.bow_vectorizer =pickle.load(open('bow_vectorizer.pkl', 'rb'))
        self.tfreq_vectorizer =pickle.load(open('tfreq_vectorizer.pkl', 'rb'))
        self.tfidf_vectorizer =pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    def load_test_data(list_headlines, body, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer):
        test_set = []
        body_bow = self.bow_vectorizer.transform([body]).toarray()
        body_tf = self.tfreq_vectorizer.transform(body_bow).toarray()[0].reshape(1, -1)
        body_tfidf = self.tfidf_vectorizer.transform([body]).toarray().reshape(1, -1)
        for headline in list_headlines:
            head_bow = self.bow_vectorizer.transform([headline]).toarray()
            head_tf = self.tfreq_vectorizer.transform(head_bow).toarray()[0].reshape(1, -1)
            head_tfidf = self.tfidf_vectorizer.transform([headline]).toarray().reshape(1, -1)
            
            tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)
            feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
            test_set.append(feat_vec)
        return np.array(test_set)

    def predict(self, list_headlines, body):
        test_set= load_test_data(list_headlines, body, word2idx)
        pred = self.model.predict(test_Set)
        return np.argmax(pred, axis = 1)