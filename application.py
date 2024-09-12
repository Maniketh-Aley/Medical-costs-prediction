from flask import Flask,request,jsonify,render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

#load the pickle files
lasso_model= pickle.load(open('models/lasso.pkl', 'rb'))
scaler_model= pickle.load(open('models/scaler.pkl', 'rb'))

#Display the Home Page
@app.route("/")
def index():
    return render_template("index.html")

#Dispaly the form the prediction
@app.route('/Predictdata', methods=['GET', 'POST'])
def predict_charges():
    if request.method == 'POST':
        # Get form data from user input
        age = int(request.form['age'])
        sex = int(request.form['sex'])  # Assuming sex is 0 or 1 (0 = Female, 1 = Male)
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])  # Assuming smoker is 0 or 1 (0 = Non-smoker, 1 = Smoker)
        region_northeast = int(request.form['region_northeast'])
        region_northwest = int(request.form['region_northwest'])
        region_southeast = int(request.form['region_southeast'])
        region_southwest = int(request.form['region_southwest'])

        new_data_scaled= scaler_model.transform([[age, sex, bmi, children, smoker, region_northeast, region_northwest, region_southeast, region_southwest]])
        result= lasso_model.predict(new_data_scaled)

        return render_template("home.html",results=result[0])
    else:
        return render_template("home.html")

if __name__=="__main__":
    app.run(host="0.0.0.0", port=8080) # default port is 5000, but can be changeable to any availabl free ports like 8080,8000
