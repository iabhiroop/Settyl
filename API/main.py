from fastapi import FastAPI
from fastapi import FastAPI
import pickle
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder
from Services.text_preprocess import pre_process
from Route import tensnn, nabayes

app = FastAPI()
app.include_router(tensnn.router)
app.include_router(nabayes.router)
with open('../random_forest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open('../label_encoder1.pkl', 'rb') as file:
    label_encoder1 = pickle.load(file)
with open('../label_encoder2.pkl', 'rb') as file:
    label_encoder2 = pickle.load(file)

@app.post("/")
def process_string(input_data: dict):
    input_string = str(input_data.get("string"))
    processed_string = pre_process(input_string)
    input_string = label_encoder1.transform([processed_string])
    prediction = np.argmax(loaded_model.predict([input_string]))
    prediction = str(label_encoder2.inverse_transform(loaded_model.predict([input_string]))[0])
    return {"prediction": prediction}