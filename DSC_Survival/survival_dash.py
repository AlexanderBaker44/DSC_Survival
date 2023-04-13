import streamlit as st
import lifelines
import pickle as pkl
import lifelines.datasets as dt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
st.set_option('deprecation.showPyplotGlobalUse', False)


#load in the dataset
df = dt.load_psychiatric_patients()

#instantiate the encoder
le_s = LabelEncoder()
#transform
sex_lis = le_s.fit_transform(df['sex'])
#put in data
df['sex'] = sex_lis

#write the title
st.header('Survival Dash')

#write some information
st.subheader('Display of data')
#write some values to the dash
st.table(df.head())

#open the file
file_waft = open('models/waft_model.pkl', 'rb')
#set the model equal to a var
waft_model = pkl.load(file_waft)
#close the file
file_waft.close()

#write some information
st.subheader('Display Unique Survival Curve')
#get min value
min_val = min(df['Age'])
#get max value
max_val = max(df['Age'])
#age slider widget
age_value = st.slider(label = 'select age',min_value = min_val,max_value = max_val)
#gender radio widget
gender = st.radio(label = 'select gender',options = list(set(df['sex'])))
#get time input
time_input = st.number_input(label='predict survival probability at time',step = 1)

#create input
inp = np.array([age_value,gender])
#reshape the array
input_surv = inp.reshape(1,-1)
#dataframe creation
d_in = pd.DataFrame(input_surv)
#add column names
d_in.columns = ['Age','sex']
#get curve
curve = waft_model.predict_survival_function(d_in)
#plot curve
curve.plot()
#display on dash
st.pyplot()
#get the prediction
prediction = waft_model.predict_survival_function(d_in,times=time_input)
#write intro
st.subheader('Get Prediction')
#process the prediction from the raw output
prediction_value = round(prediction.values[0][0],3)
#display the output
st.write(f'The probability of survival at time {time_input} is {prediction_value}.')
