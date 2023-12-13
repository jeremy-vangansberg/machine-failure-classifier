import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from transformers import TFCamembertModel, CamembertTokenizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import pickle

def load(model_path="model/saved_model/nlp_nb_class_15_2023-11-23 12_22_45.662133.h5") :
## Loading model and tokenizer
    with custom_object_scope({'TFCamembertModel': TFCamembertModel}):
        model = load_model(model_path)
        tokenizer = CamembertTokenizer("model/saved_model/tokenizer_model.model")
    return model, tokenizer

def label_encoder_custom(path="model/saved_model/labelencoder.pkl"):
    # Création de l'encodeur
    with open(path, 'rb') as file:
    # Load the LabelEncoder object from the file
        label_encoder = pickle.load(file)
    # Entraînement de l'encodeur et transformation des labels
    return label_encoder

def encode_texts(texts, tokenizer, max_seq_length):
    input_ids = []
    encoded = tokenizer(texts,max_length=max_seq_length, padding='max_length', truncation=False, return_tensors='tf',add_special_tokens=True)
    input_ids.append([encoded["input_ids"],encoded["attention_mask"]])
    return input_ids

def prediction(model, max_seq_length, tokenizer, *args) :
    texts = list(*args)
    to_test = encode_texts(texts=texts, tokenizer=tokenizer, max_seq_length=max_seq_length)
    proba = model.predict(to_test)
    indexes = np.argmax(proba, axis=1)
    max_proba = np.max(proba, axis=1)
    le = label_encoder_custom()
    return le.inverse_transform(indexes), max_proba

def predict_pipeline(to_predict:list, model, tokenizer, max_seq_length=40):
    preds, proba = prediction(model, max_seq_length, tokenizer, to_predict)
    return preds, proba