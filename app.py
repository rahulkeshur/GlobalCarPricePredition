import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
##  Load the model
model = pickle.load(open('forestModel.pkl','rb'))
encoders = pickle.load(open("encoders.pkl","rb"))
columns = pickle.load(open("columns.pkl","rb"))

def prepare_input(input_data,columns,encoders):
    result = []

    for col in columns:
        val = input_data[col]

        if col in encoders:
            val = encoders[col].transform([val])[0]
    
        result.append(val)
    return np.array(result).reshape(1,-1)


@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict_api',methods=['POST'])
def predict_api():
    user_input= request.json['user_input']
    print(user_input)
    data = prepare_input(user_input,columns,encoders)
    prediction = model.predict(data)
    print(prediction[0])
    return jsonify(prediction[0])

@app.route('/predict',methods = ['POST'])
def predict():
    data = request.form.to_dict()
    
    final_input = prepare_input(data,columns,encoders)
    print(final_input)
    output=model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The car price rediction is {}".format(output))



if __name__=="__main__":
    app.run(debug=True)
