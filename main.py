import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('rmse.txt', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')
#     return render_template("index.html")
@app.route('/predict')
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # int_features = [int(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)

    output = model

    return render_template('index.html', prediction_text='RMSE Score should be {}'.format(output))

# app.run(debug = True)
if __name__ == "__main__":
    app.run(debug=True)