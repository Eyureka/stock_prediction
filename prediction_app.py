from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import math

app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
    
    return render_template("homepage.html", pred_dict = None)



@app.route("/predict", methods=["get"])
def predict():

    weeks = 0
    try:
        weeks = math.ceil(float(request.args.get("weeks")))
    except:
        weeks = 0
    
    future_period = 1 if weeks<=0 else weeks
    
    current_date = datetime.today()
    current_date = datetime.strftime(current_date, "%Y-%m-%d")
    
    current_week = pd.date_range(start=current_date, periods=1, freq="W")[0]
    prev_trained = pickle.load(open("prev_trained_week.pkl",'rb'))
    test_size_week = (current_week-prev_trained).days//7
    
    date_list = pd.date_range(start=current_date, periods=future_period, freq="W").to_pydatetime().tolist()
    date_list = [datetime.strftime(date, "%Y-%m-%d") for date in date_list]
    
    model_filename = 'stock_prediction_model.pkl'    
    stock_prediction_model = pickle.load(open(model_filename,'rb'))
    predictions = np.round(stock_prediction_model.predict(n_periods=test_size_week + future_period), 2).tolist()[test_size_week:]
    
    dic = dict(zip(date_list,predictions))
    
    #return jsonify(dic)
    
    return render_template("homepage.html", pred_dict = dic)
    
    
    
if __name__ == "__main__":
    app.run(debug=True)
    