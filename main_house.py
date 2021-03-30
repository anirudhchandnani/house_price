from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame
app = Flask(__name__)
model = pickle.load(open('house_price_adv_xgb.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index_house.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():

    if request.method == 'POST':
        OverallQual = int(request.form['OverallQual'])
        YearBuilt=int(request.form['YearBuilt'])
        ExterQual=int(request.form['ExterQual'])
        #Kms_Driven2=np.log(Kms_Driven)
        GrLivArea=int(request.form['GrLivArea'])
        GarageCars=int(request.form['GarageCars'])
       
        lst = [[OverallQual, YearBuilt, ExterQual, GrLivArea, GarageCars]]
        X = DataFrame(lst, columns=['OverallQual', 'YearBuilt', 'ExterQual', 'GrLivArea', 'GarageCars'])
        prediction=model.predict(X)
        output=round(prediction[0],2)
        if output<0:
            return render_template('index_house.html',prediction_texts="Sorry you cannot sell this car")
        else:
            return render_template('index_house.html',prediction_text="You Can Sell The House at {}".format(output))
    else:
        return render_template('index_house.html')

if __name__=="__main__":
    app.run(debug=True)

