import streamlit as st
from predict_page import show_prediction
from eda_page import show_explore


page = st.sidebar.selectbox("Explore or Predict", ("Predict", "Exploratory Data Analysis"))

if page == "Predict":
    show_prediction()
else:
    show_explore()