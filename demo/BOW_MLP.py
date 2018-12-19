import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keras.models import load_model
import pickle


class Model:
    def load_model(self, model_file):
        model_location = "../trained_models/BOW_MLP/"
        self.model = load_model(model_location+model_file)
        self.bow_vectorizer =pickle.load(open(model_location+'bow_vectorizer.pkl', 'rb'))
        self.tfreq_vectorizer =pickle.load(open(model_location+'tfreq_vectorizer.pkl', 'rb'))
        self.tfidf_vectorizer =pickle.load(open(model_location+'tfidf_vectorizer.pkl', 'rb'))

    def load_test_data(list_headlines, body):
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
        test_set = load_test_data(list_headlines, body, word2idx)
        pred = self.model.predict(test_Set)
        return np.argmax(pred, axis = 1)
