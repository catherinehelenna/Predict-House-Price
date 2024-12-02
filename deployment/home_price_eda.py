import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from PIL import Image

def eda_main():
    st.subheader("Exploratory Data Analysis (EDA)")
    st.write("This section will reveal the patterns in the dataset.")

    # Load the dataset
    X_train_log_modified = pd.read_csv("X_train_format.csv", na_values=[], keep_default_na=False)
    y_train_log = pd.read_csv("y_train_log.csv")
    y_train_ori = pd.read_csv("y_train_ori.csv")
    # Assumes the rows in both dataframes are in the same order.  Dangerous if not true.
    X_train_log_modified['log_SalePrice'] = y_train_log.iloc[:,1]
    X_train_log_modified['SalePrice'] = y_train_ori.iloc[:,1]

    # Drop the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in X_train_log_modified.columns:
        combined_df = X_train_log_modified.drop(columns=['Unnamed: 0'])

    # Display first 5 rows of the data
    st.write("Here's a preview of the first 5 rows:")
    st.write(combined_df.head())

    st.write("Initially, the sale price distribution is severely rightly skewed. Applying log-10 transformation improved the distribution's normality.")
    st.write("D'Agostino's K-squared test resulst were compared between the original and the log-transformed sale prices, revealing increased p-value from 7.719e-133 to 3.162e-06 suggesting closer to normality.")
    st.write("Although it's still not perfectly normally distributed, the Q-Q plots below suggested most of the log-transformed sale prices follow normal distribution (red line).")

    # Outer columns (two columns)
    col1, col2 = st.columns(2)

    # Row 1
    with col1:
        st.subheader("Sale Price Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data=combined_df, x="SalePrice", kde=True, ax=ax)  # Example plot
        st.pyplot(fig)

    with col2:
        st.subheader("Log-Sale Price Distribution")
        fig, ax = plt.subplots()
        sns.histplot(data=combined_df, x="log_SalePrice", ax=ax)  # Example plot
        st.pyplot(fig)

    # Row 2
    with col1:
        st.subheader("Q-Q Plot for Sale Price")
        fig, ax = plt.subplots()
        stats.probplot(combined_df['SalePrice'], dist="norm", plot=plt)
        st.pyplot(fig)

    with col2:
        st.subheader("Q-Q Plot for Log-Sale Price ")
        fig, ax = plt.subplots()
        stats.probplot(combined_df['log_SalePrice'], dist="norm", plot=plt)
        st.pyplot(fig)

    # Row 3
    with col1:
        st.subheader("Residual Result of Lasso Regression Model")
        fig, ax = plt.subplots()
        # Load image into a PIL Image object
        image_lasso = Image.open("residual_lasso.png")
        # Display image
        st.image(image_lasso, caption="Randomly distributed residual along the zero line.", use_container_width=True)

    with col2:
        st.subheader("Q-Q Plot Residual Plot of Lasso Regression Model ")
        fig, ax = plt.subplots()
        image_lasso = Image.open("QQ residual plot lasso.png")
        # Display image
        st.image(image_lasso, caption="Most of the residuals follow the normal distribution with some outliers.", use_container_width=True)
      
    

# Run the app
if __name__ == "__main__":
    eda_main()