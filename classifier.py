import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from preprocessor import TextCleaner

print('LOADING VOCABULARY')
with open('model/vocabulary', 'rb') as model_vocab:
    vocabulary = pickle.load(model_vocab)

vectorizer = TfidfVectorizer(vocabulary=vocabulary)

print('LOADING PRE TRAINED MODEL')
with open('model/sentiment_model', 'rb') as model_clf:
    model = pickle.load(model_clf)
print('MODEL SUCCESSFULLY LOADED')


class Classifier:
    def clean_text(self, raw_text):
        stc = TextCleaner()
        return stc.cleanText(raw_text)

    def vectorize_text(self, cleaned_text):
        return vectorizer.fit_transform(cleaned_text)

    def predict_text(self, transform_text):
        return round(np.average(model.predict(transform_text)), 2)
