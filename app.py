# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:31:48 2020

@author: Rishabh Pagaria
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    # For rendering results on HTML GUI
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0][0],2)
    
    
    return render_template('index.html', prediction_text = "Score Generated for no. of hours you have studied : {}".format(output))

if __name__ == "__main__":
    app.run(debug = True)
