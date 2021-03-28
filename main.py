from typing import Optional
import uvicorn
from fastapi import FastAPI
import classy
import pandas as pd
from flask import Flask
from flask_restful import Api, Resource, reqparse
import joblib
import numpy as np
from pydantic import BaseModel

df = pd.read_csv('try_data/winequality-red.csv')
df = df.dropna(axis=0)

#Create model
target_column = 'quality'
main_model = classy.MainTensor(df,target_column)
iterations = 1
main_model.train(verb=0, nodes=8, epochs=iterations)


app = FastAPI()

class WineInputs(BaseModel):

    fixed_acidity:float
    volatile_acidity:float
    citric_acid:float
    residual_sugar:float
    chlorides:float
    free_sulfur_dioxide:float
    total_sulfur_dioxide:float
    density:float
    pH:float
    sulphates:float
    alcohol:float


@app.post('/predict')
def predict_quality(iris: WineInputs):
    
    iris = iris.dict()
    correct_x = iris
    print(correct_x)
    prediction = float(str(main_model.predict_dict(dic=correct_x))[2:-2])
    
    return {'prediction': prediction}
'''
if __name__ == '__main__':
    uvicorn.run(app,host='0.0.0.0',port=5000)
'''
