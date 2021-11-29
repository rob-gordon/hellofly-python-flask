from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
from os import path

app = Flask(__name__)
APP_ROOT = path.dirname(path.abspath(__file__))


@app.route('/')
@app.route('/<name>')
def hello(name=None):
    return render_template('hello.html', name=name)


@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    query = pd.get_dummies(query_df)

    svr = joblib.load(path.join(APP_ROOT, 'svr2.pkl'))
    prediction = svr.predict(query)
    return jsonify({'prediction': list(prediction)})
