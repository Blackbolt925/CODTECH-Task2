import pandas as pd
import numpy as np 
import tensorflow as tf 
import re
from numpy import array

from keras.preprocessing.text import one_hot, Tokenizer
from keras.models import Sequential, load_model
from keras.layers import LSTM
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, GlobalMaxPooling1D, Embedding, Conv1D, LSTM
from sklearn.model_selection import train_test_split
from flask import Flask, request, jsonify, render_template
from keras_preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from keras_preprocessing.text import tokenizer_from_json
import io
import json

stopwords_list = set(stopwords.words('english'))
maxlen = 100

from tensorflow.keras.models import load_model

# Replace the path with your actual saved model directory
loaded_model = load_model(f"./sentiment_IMDb_model_acc_{round(scores[1], 3)}")

with open('tokenizer.json') as f:
    data=json.load(f)
    loaded_tokenizer = tokenizer_from_json(data)

app = Flask(__name__)

from preprocessing_func import Preprocess
custom= Preprocess()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    query_asis = [str(x) for x in request.form.values()]

    
    
    query_processed_list = []
    for query in query_asis:
        query_processed = custom.preprocess_text(query)
        query_processed_list.append(query_processed)
        
    
    query_tokenized = loaded_tokenizer.texts_to_sequences(query_processed_list)
    
    
    query_padded = pad_sequences(query_tokenized, padding='post', maxlen=maxlen)
    
    
    query_sentiments = loaded_model(query_padded)
    

    if query_sentiments[0][0]>0.5:
        return render_template('index.html', prediction_text=f"Positive Review with probable IMDb rating as: {np.round(query_sentiments[0][0]*10,1)}")
    else:
        return render_template('index.html', prediction_text=f"Negative Review with probable IMDb rating as: {np.round(query_sentiments[0][0]*10,1)}")

if __name__ == '__main__':
    app.run(debug=True)