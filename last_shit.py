#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:05:19 2024

@author: noob
"""
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load and preprocess data
data = []
with open("cars_all.txt", "r") as file:
    for line in file:
        data.append(json.loads(line.strip()))

df = pd.DataFrame(data)
df['brand'] = df['brand'].apply(lambda x: x.split('||')[1].strip())
df['model'] = df['model'].apply(lambda x: x.split('||')[0].strip() if len(x.split('||')) == 1 else x.split('||')[1].strip())
df['price'] = df['price'].apply(lambda x: int(x.replace("جنيه","").replace(",","").strip()))

df['fuel_type'].value_counts()
df['fuel_type'] = df['fuel_type'].apply(lambda x: x.strip())
df = df[df['fuel_type'].isin(["أتوماتيك‎","مانيوال"])]

df['km'] = df['km'].apply(lambda x: x.strip() if "كم" in x else None)
df = df.dropna(subset=['km'])
df['km'] = df['km'].apply(lambda x: x.replace("كم","").replace(",",""))
df['km'] = df['km'].apply(lambda x: int(x))
df['year'] = df['year'].apply(lambda x: int(x) if x.isdigit() else None)
df = df.dropna(subset=['year'])
df = df[df['km'] > 50000]
df = df[df['price'] != 0]

label_encoders = {}
for column in ['brand', 'model', 'fuel_type']:
    le = LabelEncoder()
    df[f'{column}_encode'] = le.fit_transform(df[column])
    label_encoders[column] = le
    
df_level3 = df[['brand_encode', 'model_encode', 'year', 'km', 'fuel_type_encode', 'price']]

from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame() 
vif_data["feature"] = df_level3.columns 
  
# calculating VIF for each feature 
vif_data["VIF"] = [variance_inflation_factor(df_level3.values, i) 
                          for i in range(len(df_level3.columns))] 
print(vif_data)

Numerics=LabelEncoder()

df_level3['brand_model_and_year_combined'] = df_level3['brand_encode'].astype(str) + '_' + df_level3['model_encode'].astype(str) + '_' + df_level3['year'].astype(str)
df_level3['brand_model_and_year_combined_encoded'] = Numerics.fit_transform(df_level3['brand_model_and_year_combined'])
df_level3 = df_level3.drop(columns=['brand_encode','model_encode', 'year','brand_model_and_year_combined'])

relation = df_level3.corr()

df_level3 = df_level3[['brand_model_and_year_combined_encoded', 'km', 'fuel_type_encode', 'price']]
X = df_level3.drop("price", axis=1).values
y = df_level3["price"].values


model = RandomForestRegressor(n_estimators=159, random_state=42)

# Cross-validation
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f'Cross-validated R2 scores: {scores}')
print(f'Average R2 score: {scores.mean()}')

# Train-test split for final evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'R2 Score: {r2}')

# Retrieve feature importances
importances = model.feature_importances_

# Create a DataFrame for better visualization
features = [f'Feature {i}' for i in range(X.shape[1])]
importance_df = pd.DataFrame({
    'Feature': df_level3.iloc[:, :-1].columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)


import pickle

with open('used_car_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model saved to 'used_car_model.pkl'")

# in UI
import pickle

with open('used_car_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

print("Model loaded successfully")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(relation.index, abs(relation.values))
plt.xlabel('Importance')
plt.title('Feature Importances (Before)')
plt.gca().invert_yaxis()
plt.show()

plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importances  (After)')
plt.gca().invert_yaxis()
plt.show()


# brand_df = df[['brand','brand_encode']]
# brand_df = brand_df.drop_duplicates()
# brand_df.to_csv('brands.csv')

# model_df = df[['model','model_encode']]
# model_df = model_df.drop_duplicates()
# model_df.to_csv('model.csv')


# fu_df = df[['fuel_type','fuel_type_encode']]
# fu_df = fu_df.drop_duplicates()
# fu_df.to_csv('fu.csv')

# df[['year']].to_csv('year.csv')



# from sklearn.model_selection import GridSearchCV

# # Define the parameter grid
# param_grid = {
#     'n_estimators': [50, 100, 200, 300 ,159],
#     'random_state': [0,1,42,100] # Fixed values for demonstration; typically more extensive
# }

# # Initialize the model
# model = RandomForestRegressor()

# # Initialize GridSearchCV
# grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1000)

# # Fit GridSearchCV
# grid_search.fit(X_train, y_train)

# # Output the best parameters and score
# print(f"Best parameters: {grid_search.best_params_}")
# print(f"Best R2 score: {grid_search.best_score_}")
