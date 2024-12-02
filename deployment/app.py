import streamlit as st
from home_price_eda import eda_main
from home_price_prediction import prediction_main

# Title of the app
st.title("🏡 Home Price Predictor")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a section:", ["Home", "Exploratory Data Analysis", "Prediction"])

# Home Section
if page == "Home":
    # Text description
    st.write("Welcome to the Home Price Predictor!")
    st.write("""
    This application helps you predict home prices based on various features and perform exploratory data analysis 
    to understand the dataset better.
    """)

# EDA Section
elif page == "Exploratory Data Analysis":
    eda_main()  # Call the EDA module's main function
    

# Prediction Section
elif page == "Prediction":
    prediction_main()  # Call the prediction module's main function