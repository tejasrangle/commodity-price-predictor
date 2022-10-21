from flask import *
import pandas as pd
import pickle
import numpy as np


app = Flask(__name__)

df = pd.read_csv("cleaned_car_data.csv")
model = pickle.load(open("LinearRegressionModel.pkl","rb"))
df_bike = pd.read_csv("cleaned_bike_data.csv")
model_bike = pickle.load(open("LinearRegressionModel_bike.pkl","rb"))

@app.route("/")
def home():
    return render_template("home_car_bike.html")

@app.route("/action",methods=["POST"])
def commodity():
    commodity=request.form["commodity"]
    if commodity =="car":
        companies = sorted(df["company"].unique())
        car_models = sorted(df["name"].unique())
        year = sorted(df["year"].unique())
        fuel_type = sorted(df["fuel_type"].unique())
        return render_template("car.html",companies=companies,car_models=car_models,year=year,fuel_type=fuel_type)
    else:
        brands = sorted(df_bike["brand"].unique())
        bike_names = sorted(df_bike["bike_name"].unique())
        owner = sorted(df_bike["owner"].unique())
        return render_template("bike.html",brands=brands,bike_names=bike_names,owner=owner)

@app.route("/action_car",methods=["POST"])
def action_car():
    company=request.form["company"]
    car_model=request.form["car_model"]
    car_year=int(request.form["car_year"])
    fuel_type=request.form["fuel_type"]
    Kilometer=int(request.form["Kilometer"])
    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],data=np.array([car_model,company,car_year,Kilometer,fuel_type]).reshape(1, 5)))
    if prediction>0:
        return render_template("result_car.html",prediction=np.round(prediction[0],2))
    else:
        return render_template("result_car.html",prediction="Scrap")

@app.route("/action_bike",methods=["POST"])
def action_bike():
    brand=request.form["brand"]
    bike_name=request.form["bike_name"]
    age=int(request.form["age"])
    owner=request.form["owner"]
    Kilometer=request.form["Kilometer"]
    prediction=model_bike.predict(pd.DataFrame(columns=['bike_name', 'kms_driven', 'owner', 'age', 'brand'],data=np.array([bike_name,Kilometer,owner,age,brand]).reshape(1, 5)))
    if prediction>0:
        return render_template("result_bike.html",prediction=np.round(prediction[0],2))
    else:
        return render_template("result_bike.html",prediction="Scrap")

if __name__ =="__main__":
    app.run(debug=True)