from flask import Flask, request, render_template
from joblib import load
import pickle 


import pandas as pd
df=pd.read_csv('mydatacars4.csv')
df= df.drop(["Selling_Price","Kms_Driven"],axis=1)
dg = pd.DataFrame()
dg["Brand"] = df["Car_Name"].apply(lambda x : x.split()[0])
df= dg.join(df)
del dg
#!!!!!!!!!!!!!!!!!!!!!!!!!!!
from sklearn.preprocessing import LabelEncoder
Numerics=LabelEncoder()
df['car_NV']=Numerics.fit_transform(df['Car_Name'])
df['brand_NV']=Numerics.fit_transform(df['Brand'])
#
cars = pd.DataFrame(df['Car_Name'].unique(),columns=(["Car_Name"]))
cars['car_NV']=Numerics.fit_transform(cars['Car_Name'])
#!!!!!!!!!!!!!!!!!!!!!!!!!!!

from markupsafe import Markup
options = ""
for x in range(0,len(cars)):
    options+= Markup('<option value=\"'+str(cars.iloc[x].car_NV)+'\">'+str(cars.iloc[x].Car_Name)+'</option>')
del cars
#!!!!!!!!!!!!!!!!!!!!!!!!!!!


#### TEST
app = Flask(__name__)

model = load('ml.pkl') # or load('ml.pkl') if error
@app.route('/', methods=["POST", 'GET'])
def home():
    if request.method == "POST":
        dia = [[float(x) for x in request.form.values()]]
        #??????????????????????????
        row = df.loc[(df['car_NV'] == float(str(dia[0][0]))) & (df['Year'] == float(str(dia[0][1])))]
        
        if len(row) == 0 :
            row =  df.loc[(df['car_NV'] == float(str(dia[0][0])))]
            d_Year = 2000
            for myindex in range(len(row)):
                if float(str(dia[0][1])) <= float(row.iloc[myindex].Year) :
                    d_Year = float(row.iloc[myindex].Year)
                    break
                else:
                    if myindex == len(row)-1:
                        d_Year = float(row.iloc[myindex].Year)   
            row = df.loc[(df['car_NV'] == float(str(dia[0][0]))) & (df['Year'] == d_Year)]
            row = row.head(1)
        else:
            row = row.head(1)
            
        row = row.drop(["Fuel_Type","Transmission","Car_Name","car_NV","Year","Brand"],axis=1)
        dia[0].append(float(row.Mileage))
        dia[0].append(float(row.Engine))
        dia[0].append(float(row.Power))        
        dia[0].append(float(row.Seats))        
        dia[0].append(float(row.brand_NV))        
        #??????????????????????????
        predict = model.predict(dia)
        return render_template('answer.html', predict=predict)
        #return render_template('answer.html', predict=dia)
    else:
        return render_template('index.html',PP=options)

if __name__ == '__main__':
    app.run()