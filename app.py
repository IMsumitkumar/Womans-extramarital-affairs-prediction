import streamlit as st
import pandas as pd
import numpy as np
import pickle
from scipy.sparse import hstack
import os

import warnings
warnings.filterwarnings('ignore')

# changing page main title and main icon(logo)
PAGE_CONFIG = {"page_title":"Affairs prediction", "page_icon":":cancer:", "layout":"centered"}
st.set_page_config(**PAGE_CONFIG)   

st.sidebar.text("Created on Sat, Feb 2 2021")
st.sidebar.markdown("**@author:Sumit Kumar** :monkey_face:")
st.sidebar.markdown("[My Github](https://github.com/IMsumitkumar) :penguin:")
st.sidebar.markdown("[findingdata.ml](https://www.findingdata.ml/) :spider_web:")
st.sidebar.markdown("coded with :heart:")

# sidebar header
st.sidebar.subheader("Affairs prediction")

st.subheader("Prediction of extramarital affair of woman")
st.title("")

occupation = ('student','farming/semiskilled/unskilled','white collar','teacher/nurse/writer/technician/skilled','managerial/business', 'professional with advanced degree')
rate_marriage = ('very poor', 'poor', 'OK', 'good', 'very good')
religious = ('not religious', 'religious', 'some religious', 'strongly religious')
educ = ('grade school', 'high school', 'some college', 'college graduate', 'some graduate school', 'advanced degree')

std = pickle.load(open('models/std_norm.sav', 'rb'))
model = pickle.load(open('models/logistic_model.pkl', 'rb')) 

# st.image("https://i.imgur.com/wdOwQKj.png", width=700)
st.sidebar.image("https://i.imgur.com/wdOwQKj.png", width=700)


five, six, seven = st.beta_columns(3)
age = five.number_input('Enter your Age')
yr_married = six.number_input("year's of marriage")
children = seven.number_input("Number of childeren's do you have?")

one, two = st.beta_columns(2)
wife_occupation = one.selectbox(
"wife occupation?", occupation)

husband_occupation = two.selectbox(
"Husband Occupation?", occupation)

three, four = st.beta_columns(2)
marriage_rate = three.selectbox(
"Rate on your marriage state?", rate_marriage)

religion = four.selectbox(
"Rate on your religious state?", religious)

education = st.selectbox(
"what is your education level?", educ)


if wife_occupation == 'student':
    occ_2 = 0
    occ_3 = 0
    occ_4 = 0
    occ_5 = 0
    occ_6 = 0
elif wife_occupation == 'farming/semiskilled/unskilled':
    occ_2 = 1
    occ_3 = 0
    occ_4 = 0
    occ_5 = 0
    occ_6 = 0
elif wife_occupation == 'white collar':
    occ_2 = 0
    occ_3 = 1
    occ_4 = 0
    occ_5 = 0
    occ_6 = 0
elif wife_occupation == 'teacher/nurse/writer/technician/skilled':
    occ_2 = 0
    occ_3 = 0
    occ_4 = 1
    occ_5 = 0
    occ_6 = 0
elif wife_occupation == 'managerial/business':
    occ_2 = 0
    occ_3 = 0
    occ_4 = 0
    occ_5 = 1
    occ_6 = 0
elif wife_occupation == 'professional with advanced degree':
    occ_2 = 0
    occ_3 = 0
    occ_4 = 0
    occ_5 = 0
    occ_6 = 1


if husband_occupation == 'student':
    occ_husb_2 = 0
    occ_husb_3 = 0
    occ_husb_4 = 0
    occ_husb_5 = 0
    occ_husb_6 = 0
elif husband_occupation == 'farming/semiskilled/unskilled':
    occ_husb_2 = 1
    occ_husb_3 = 0
    occ_husb_4 = 0
    occ_husb_5 = 0
    occ_husb_6 = 0
elif husband_occupation == 'white collar':
    occ_husb_2 = 0
    occ_husb_3 = 1
    occ_husb_4 = 0
    occ_husb_5 = 0
    occ_husb_6 = 0
elif husband_occupation == 'teacher/nurse/writer/technician/skilled':
    occ_husb_2 = 0
    occ_husb_3 = 0
    occ_husb_4 = 1
    occ_husb_5 = 0
    occ_husb_6 = 0
elif husband_occupation == 'managerial/business':
    occ_husb_2 = 0
    occ_husb_3 = 0
    occ_husb_4 = 0
    occ_husb_5 = 1
    occ_husb_6 = 0
elif husband_occupation == 'professional with advanced degree':
    occ_husb_2 = 0
    occ_husb_3 = 0
    occ_husb_4 = 0
    occ_husb_5 = 0
    occ_husb_6 = 1


if marriage_rate == 'very poor':
    marriage_rate = 1
elif marriage_rate == 'poor':
    marriage_rate = 2
elif marriage_rate == 'OK':
    marriage_rate = 3
elif marriage_rate == 'good':
    marriage_rate = 4
elif marriage_rate == 'very good':
    marriage_rate = 5


if religion == 'not religious':
    religion = 1
elif religion == 'religious':
    religion = 2
elif religion == 'some religious':
    religion = 3
elif religion == 'strongly religious':
    religion = 4

if education == 'grade school':
    education = 0 
elif education == 'high school':
    education = 1
elif education == 'some college':
    education = 2 
elif education == 'college graduate':
    education = 3 
elif education == 'some graduate school':
    education = 4 
elif education == 'advanced degree':
    education = 5 

query_vector = [[occ_2, occ_3, occ_4, occ_5, occ_6, occ_husb_2, occ_husb_3, occ_husb_4, occ_husb_5,\
                    occ_husb_6, marriage_rate, age, yr_married, children, religion, education]]

if st.button("Predict"):
    predicted_cls = model.predict(std.transform(query_vector))
    st.markdown("Predicted Class")
    st.success(predicted_cls[0])
    predicted_probas = np.round(model.predict_proba(std.transform(query_vector)), 3)
    st.markdown("Predicted class probabilities : Not having Affair   :"+ str(predicted_probas[0][0]))
    st.markdown("Predicted class probabilities : Having Affair   :"+ str(predicted_probas[0][1]))
