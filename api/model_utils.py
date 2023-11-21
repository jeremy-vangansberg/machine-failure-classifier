import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
from transformers import TFCamembertModel, CamembertTokenizer
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def load(model_path="nlp_14_nb_class_2023-11-21 15_14_07.250352") :
## Loading model and tokenizer
    with custom_object_scope({'TFCamembertModel': TFCamembertModel}):
        model = load_model(model_path)
        tokenizer = CamembertTokenizer("model/saved_model/tokenizer_model.model")
    return model, tokenizer

def label_extractor(path="model/labels.csv", col='labels'):
    df = pd.read_csv(path)
    return df[col]

def label_encoder():
    # Création de l'encodeur
    le = LabelEncoder()
    y = label_extractor()
    # Entraînement de l'encodeur et transformation des labels
    le.fit_transform(y)
    return le


def encode_texts(texts, tokenizer, max_seq_length):
    input_ids = []
    for text in texts:
        encoded = tokenizer(text, return_tensors="tf", truncation=True, padding="max_length", max_length=max_seq_length)
        input_ids.append(encoded["input_ids"][0])
    return np.array(input_ids)

def prediction(model, max_seq_length, tokenizerf, *args) :
    texts = list(*args)
    to_test = encode_texts(texts=texts, tokenizer=tokenizerf, max_seq_length=max_seq_length)
    proba = model.predict(to_test)
    indexes = np.argmax(proba, axis=1)
    le = label_encoder()
    return le.inverse_transform(indexes)

def predict_pipeline(to_predict:list, max_seq_length=32):
    model, tokenizer = load()
    preds = prediction(model, max_seq_length, tokenizer, to_predict)

    return preds




