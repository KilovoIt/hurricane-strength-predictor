# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output, State
import pandas as pd
#
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier 
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import dash_daq as daq


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
#LOADING DATA
hurr = pd.read_csv(r"X:\LambdaSchool\hurricanes-final.csv")
hurr.drop("Unnamed: 0", axis=1, inplace=True)
hurr.info()

###DATA SPLITTING

train, test = train_test_split(hurr, test_size=0.1)
target = 'USA_WIND_kts'
y = train[target]
X = train.drop(target, axis=1)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2)

###ENCODING CAT DATA
encoder = OrdinalEncoder()
X_train[['NATURE']] = encoder.fit_transform(X_train[['NATURE']])
X_val[['NATURE']] = encoder.transform(X_val[['NATURE']])

###XGBRegressor
model_boost = XGBRegressor()

param_grid ={}

xgbgs = GridSearchCV(model_boost, param_grid, cv=2)

xgbgs.fit(X_train, y_train)
#print(xgbgs.score(X_train, y_train))
#print(xgbgs.score(X_val, y_val))


# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
#df = pd.DataFrame({
#    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#    "Amount": [4, 1, 2, 2, 4, 5],
#    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
#})
#
#fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")


    #html.H1("Hurricane Strength Predictor", style={'text-align':'center'}),
    
app.layout = html.Div([

html.H1(children='Hurricane Strength Predictor', id='heading',style={'padding': 50, 'textAlign': 'center'}),



html.Div([
html.H5(children='This is a hurrciane strength estimator model. Input coordinates into the corresponding fields and wait for the result see what can be an approximate hurricane strength for this location is possible. The hurricane system find the closest point that exists in the database and predicts possible wind speed in knots. After the wind speed is calculated, hurricane cathegory is calculated as well based on Saffir-Simpson scale.',
 style={'float':'left', 'margin': 80, 'width': 1000, 'height': 'auto'})
]),



html.Div([
   #dcc.Input(
    #placeholder='Input Latitude and Longitude, comma separated',
    #type='number',
    #value='',
    #style={"height": "10", "width": 800, "margin-bottom": "auto"},
    #id='latlon-input'),


#html.Div([
#html.Button('Estimate',
#style={"height": "10", "margin-bottom": "auto", 'color': 'black', 'background-color': '#95d8fc', 'float':'right'}, 
#id='button')]),

daq.NumericInput(
        id='lat-input',
        size = 140,
        max = 90,
        min =-90,
        label='Latitude',
        labelPosition='bottom',
        value=0,
         style={'float':'left'}
    ),
    
daq.NumericInput(
        id='lon-input',
        size = 140,
        max = 180,
        min = 0,
        label='Longitude',
        labelPosition='bottom',
        value=0, 
        style={'float':'right'}
    ),

html.Div([
daq.Gauge(
  id='cat-gauge',
  label='CATEGORY',
  min=0,
  max=5),
  
daq.Gauge(
  id='speed-gauge',
  label='SUSTAINED SPEED, kts',
  min=0,
  max=195)], style = {'padding': 80})
   
    
], style={'margin': 80, "padding": 25, 'float': 'right', 'width': 450})

])


@app.callback([
    Output(component_id='cat-gauge', component_property='value'),
    Output(component_id='speed-gauge', component_property='value')],
    [Input(component_id='lat-input', component_property='value'), Input(component_id='lon-input', component_property='value')]
)
def update_output_div(input_value1, input_value2):
    found = 10000000
    for i in hurr.index:
        if found > (abs(hurr['LAT_degrees_north'][i] - input_value1 + hurr['LON_degrees_east'][i] - input_value2)):
            found = abs((hurr['LAT_degrees_north'][i] - input_value1 + hurr['LON_degrees_east'][i] - input_value2))
            coeff = i
    #t = (hurr['LAT_degrees_north'][coeff], hurr['LON_degrees_east'][coeff])
    m  =  hurr[hurr.index == coeff].drop('USA_WIND_kts', axis=1)
    m[['NATURE']] = encoder.transform(m[['NATURE']])
    k  = xgbgs.predict(m)
    if k[0] < 64:
        s = 0
    elif ((k[0] >= 64) and (k[0] < 82)):
        s = 1
    elif ((k[0] >= 82) and (k[0] < 95)):
        s = 2
    elif ((k[0] >= 95) and (k[0] < 112)):
        s = 3
    elif ((k[0] >= 112) and (k[0] < 136)):
        s = 4
    elif k[0] >= 136:
        s = 5
    return  s, k[0]

if __name__ == '__main__':
    app.run_server(debug=True)