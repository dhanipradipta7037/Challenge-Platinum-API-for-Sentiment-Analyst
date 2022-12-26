from ast import Delete, In
from os import remove
from tkinter import S
from flask import Flask, request, jsonify
from flasgger import Swagger, LazyString, LazyJSONEncoder, swag_from
import re, pandas as pd, sqlite3, json, tensorflow
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
import pickle
tensorflow.config.experimental.list_physical_devices('GPU')
import tensorflow as tf
import numpy as np


app = Flask(__name__)

app.json_encoder = LazyJSONEncoder


swagger_template = dict(
    info = {
        'title': LazyString(lambda:'API for Sentiment Analysis'),
        'version': LazyString(lambda:'1'),
        'description': LazyString(lambda:'Platinum Challenge Data Science Binar Academy')
        }, host = LazyString(lambda: request.host)
    )

swagger_config = {
        "headers":[],
        "specs":[
            {
            "endpoint":'docs',
            "route":'/docs.json'
            }
        ],
        "static_url_path":"/flasgger_static",
        "swagger_ui":True,
        "specs_route":"/docs/"
    }

swagger = Swagger(app, template=swagger_template, config=swagger_config)

# Connect dengan platinum db
db = sqlite3.connect('Platinum.db' ,  check_same_thread=False)
alay = pd.read_csv("new_kamusalay.csv", encoding = "latin_1")
alayFix = alay["fix"].to_list()
alayOri = alay["alay"].to_list()


def cleansing (txt):
    txt = str(txt).lower()

    txt = re.sub("[,]", " ,", txt)
    txt = re.sub("[.]", " .", txt)
    txt = re.sub("[?]", " ? ", txt)
    txt = re.sub("[!]", " !", txt)
    txt = re.sub("[\"]", " \"", txt)
    txt = re.sub("[\']", "", txt)
    txt = re.sub("[\\n]", " ", txt)
    txt = re.split(" ", txt)

    for word in txt :
        if word in alayOri :
            txt[txt.index(word)] = txt[txt.index(word)].replace(word, alayFix[alayOri.index(word)])

    txt = " ".join(txt)
    txt = re.sub(" ,", ",", txt)
    txt = re.sub(" \.", ".", txt)
    txt = re.sub(" \?", "?", txt)
    txt = re.sub(" !", "!", txt)
    txt = re.sub(" \"", "\"", txt)

    return txt

# API LSTM (Text)
@swag_from("Docs/lstm_input_data.yml",methods=['POST'])
@app.route('/lstm_text',methods=['POST'])
def lstm_text():
    original_text = str(request.form["lstm_text"])
    text_output = cleansing(original_text)

    loaded_model = load_model(r'lstm2.h5')

    with open('tokenizer_lstm2.json') as f:
        data = json.load(f)
        tokenizer_lstm = tokenizer_from_json(data)
    
    def pred_sentiment(string):    
        string = re.sub(r'[^a-zA-Z0-9. ]', '', string)
        string = string.lower()
        txt1 = [string]

        sekuens_x = tokenizer_lstm.texts_to_sequences(txt1)
        padded_x = pad_sequences(sekuens_x)

        classes = loaded_model.predict(padded_x, batch_size=10)

        return classes[0]

    def pred(classes):
        if classes[0] == classes.max():
            return ('Negatif')
        if classes[1] == classes.max():
            return ('Netral')
        if classes[2] == classes.max():
            return ('Positif')

    classes2 = pred_sentiment(text_output)
    classes3 = pred(classes2)

    json_response = {
      'status_code' : 200,
        'description' : "Result of Sentiment Analysis using LSTM",
        'data' : {
            'Text' : text_output ,
            'Sentiment' : classes3
        },
    }

    db.execute('create table if not exists lstm_input_data (original_text varchar(255), text varchar(255), sentiment varchar(255))')
    query_text = 'insert into lstm_input_data (original_text , text, sentiment) values (?,?,?)'
    val = (original_text,text_output,classes3)
    db.execute(query_text,val)
    db.commit()

    response_data = jsonify(json_response)
    return response_data


# API NN (Text)
@swag_from("Docs/nn_input_data.yml",methods=['POST'])
@app.route('/nn_text',methods=['POST'])
def nn_text():
    original_text = str(request.form.get('nn_text'))
    text_output = [cleansing(original_text)]

    loaded_model = pickle.load(open('feature_nn.pkl', 'rb'))

    model_NN=pickle.load(open('model_nn.pkl', 'rb'))

    text = loaded_model.transform(text_output)

    sentimen = model_NN.predict(text)[0]

    json_response = {
      'status_code' : 200,
        'description' : "Result of Sentiment Analysis using NN",
        'data' : {
            'Text' : text_output ,
            'sentiment' : sentimen
        },
    }

    query_text = 'insert into nn_input_data (original_text , text, sentiment) values (?,?,?)'
    val = (original_text,text_output[0],sentimen)
    db.execute(query_text,val)
    db.commit()

    response_data = jsonify(json_response)
    return response_data



# API LSTM (Upload Text)
@swag_from("docs/lstm_upload_data.yml", methods=['POST'])
@app.route('/lstm_upload', methods=['POST'])
def lstm_upload():
    file = request.files["lstm_upload"]
    df_csv = (pd.read_csv(file, encoding="latin-1"))
    df_csv_1 = df_csv['Tweet'].apply(cleansing)
    new_tweet = df_csv_1

    loaded_model = load_model(r'lstm2.h5')

    with open('tokenizer_lstm2.json') as f:
        data = json.load(f)
        tokenizer = tokenizer_from_json(data)


    def pred_sentiment(string):    
        sekuens_x = tokenizer.texts_to_sequences(string)
        padded_x = pad_sequences(sekuens_x)

        classes = loaded_model.predict(padded_x, batch_size=10)

        return classes[0]

    def pred(classes):
        if classes[0] == classes.max():
            return ('Negatif')
        if classes[1] == classes.max():
            return ('Netral')
        if classes[2] == classes.max():
            return ('Positif')


    for teks in new_tweet:
        ori = teks
        classes = pred_sentiment(teks)

        classes2 = pred(classes)

        db.cursor().execute("insert into Platinum_lstm (TweetOri, TweetClean, Sentimen) values(?, ?, ?)", (ori, teks, classes2))
        db.commit() 

    response_data = jsonify("SUKSES")
    return response_data

# API NN (Upload Text)
@swag_from("docs/nn_upload_data.yml", methods=['POST'])
@app.route('/nn_upload', methods=['POST'])
def nn_upload():
    file = request.files["nn_upload"]
    df_csv = (pd.read_csv(file, encoding="latin-1"))
    df_csv_1 = df_csv['Tweet']
    new_tweet = df_csv_1

    loaded_model = pickle.load(open('feature_nn.pkl', 'rb'))

    model_NN=pickle.load(open('model_nn.pkl', 'rb'))

    for teks in new_tweet:
        ori = teks
        text1 = cleansing(teks)
        text = loaded_model.transform([teks])

        sentimen = model_NN.predict(text)[0]


        db.execute('create table if not exists Platinum_nn (original_text varchar(255), text varchar(255), sentiment varchar(255))')
        query_text = 'insert into Platinum_nn (original_text , text, sentiment) values (?,?,?)'
        val = (ori,text1,sentimen)
        db.execute(query_text,val)
        db.commit()

    response_data = jsonify("SUKSES")
    return response_data


if __name__ == '__main__':
	app.run(debug=True, port=8080)