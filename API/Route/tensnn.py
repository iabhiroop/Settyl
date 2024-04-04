from venv import create
from fastapi import APIRouter
import pickle
import numpy as np
from Services.text_preprocess import pre_process
import tensorflow as tf
import keras
from keras.layers import Dense
from keras.models import Sequential

router = APIRouter(prefix='/tensor', tags=['tensnn'])
model = Sequential([
    Dense(128, activation='relu', input_shape=(1,)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(15, activation="softmax")
])
checkpoint_path = '../tensornn.weights.h5'
model.load_weights(checkpoint_path)
    
with open('../label_encoder1.pkl', 'rb') as file:
    label_encoder1 = pickle.load(file)
with open('../label_encoder2.pkl', 'rb') as file:
    label_encoder2 = pickle.load(file)
    
@router.post("/")
def process_string(input_data: dict):
    input_string = str(input_data.get("string"))
    processed_string = pre_process(input_string)
    input_string = label_encoder1.transform([processed_string])
    prediction = np.argmax(model.predict([input_string]))
    prediction = str(label_encoder2.inverse_transform([np.argmax(model.predict([input_string]))])[0])
    return {"prediction": prediction}