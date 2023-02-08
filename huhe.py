# Packages for EDA 
import pandas as pd 
import numpy as np 
import seaborn as sns 

# Data Preprocessing
from sklearn.model_selection import train_test_split
import category_encoders as ce
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PowerTransformer

# Modeling
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import ElasticNetCV
np.seterr(divide='ignore', invalid='ignore', over='ignore')

sns.set(rc={'figure.figsize': [7, 14]}, font_scale=1.2) # Standard figure size for all 

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

df = pd.read_csv("mydatacars4.csv")

if(df.duplicated().sum() == 0):
    df.drop_duplicates(inplace=True)
    
#####################################
# Custom Encoding
df["Brand"] = df["Car_Name"].apply(lambda x : x.split()[0])

df["Car_Name"] = df["Car_Name"].apply(lambda x : " ".join(x.split()[:2]))
df = df.convert_dtypes()

#.df.to_csv("Cleaned_Data.csv")


transformation = {
    "Manual":1,
    "Auto":0
}

df['Transmission'] = df['Transmission'].map(transformation)


transformation = {
    "Elec":2,
    "Gas":1,
    "Petrol":0
}

df['Fuel_Type'] = df['Fuel_Type'].map(transformation)

##################################################^^^^
# Encoding
from sklearn.preprocessing import LabelEncoder
Numerics=LabelEncoder()
new_df = df
new_df['Car_Name']=Numerics.fit_transform(df['Car_Name'])
new_df['Brand']=Numerics.fit_transform(df['Brand'])
#################

# define dataset
X, y = new_df.drop("Selling_Price",axis=1).to_numpy() , new_df["Selling_Price"].to_numpy() 
# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# !!!!!!!!!!!!!!!!!!!!!!!!

reg = LinearRegression().fit(X_train,(y_train))
print("Accurecy",round((reg.score(X_test,  (y_test))*100),2),'%')
################################
reg.predict([[2,2022,30000,0,0,9.8,3000,275,5,2]])
#############

import pickle
pickle.dump(reg, open('ml.pkl','wb'))