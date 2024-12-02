import streamlit as st
import pandas as pd
import numpy as np
import joblib

def prediction_main():
    st.header("Home Price Prediction")
    st.write("This section allows you to predict home prices based on input features.")
    st.subheader("Model specification and performance details.")
    st.markdown("""
        Model: `Lasso(alpha=0.0005, random_state=42)`
        cross-validation: R2-score = 0.892 and MSE = 0.0033
        testing: R2-score = 0.932 and MSE = $306,898,281
        Note: MSE for cross-validation is for log-price prediction.
    """)
    # Load the model with a context manager
    with open("capped_lasso_pipeline.joblib", "rb") as file:
        model = joblib.load(file)
    # Load the dataset
    X_train_log_modified = pd.read_csv("X_train_format.csv", na_values=[], keep_default_na=False)


    # Drop the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in X_train_log_modified.columns:
        X_train_log_modified = X_train_log_modified.drop(columns=['Unnamed: 0'])


    # Feature names
    # feature_names = X_train_log_modified.columns.tolist()

    # Numeric features
    final_numeric_features = X_train_log_modified.select_dtypes(include=np.number)
    final_numeric_feature_ranges = final_numeric_features.agg(['min', 'max'])

    # Categorical features
    final_categorical_features = X_train_log_modified.select_dtypes(exclude=np.number)
    final_categorical_feature_groups = final_categorical_features.apply(lambda x: x.unique().tolist())

    # UI for Numeric Features
    st.subheader("Numeric Features")
    st.write("Adjust the range for numeric features using sliders:")
    numeric_inputs = {}
    for feature in final_numeric_features.columns:
        min_value = final_numeric_feature_ranges.loc['min', feature]
        max_value = final_numeric_feature_ranges.loc['max', feature]
        numeric_inputs[feature] = st.slider(
            label=f"{feature}",
            min_value=float(min_value),
            max_value=float(max_value),
            value=float((min_value + max_value) / 2)
        )

    # UI for Categorical Features
    st.subheader("Categorical Features")
    st.write("Select a value for each categorical feature:")
    categorical_inputs = {}
    for feature in final_categorical_features.columns:
        options = final_categorical_feature_groups[feature]
        categorical_inputs[feature] = st.selectbox(
            label=f"{feature}",
            options=options
        )

    # Combine the numeric and categorical inputs
    user_inputs = {}

    # Add numeric inputs
    for feature, selected_range in numeric_inputs.items():
        # Take the midpoint of the range for simplicity
        user_inputs[feature] = np.mean(selected_range)

    # Add categorical inputs
    for feature, selected_value in categorical_inputs.items():
        user_inputs[feature] = selected_value

    # Create a DataFrame from user inputs
    user_input_df = pd.DataFrame([user_inputs])

    # Reorder the DataFrame columns to match X_train_log_modified
    user_input_df = user_input_df[X_train_log_modified.columns]

    # Display the combined and ordered DataFrame
    st.header("Combined User Input Data")
    st.write(user_input_df)

    # Predict the house price
    if model is not None:
        st.header("Predicted House Price")
        try:
            log_prediction = model.predict(user_input_df)[0]
            real_prediction = np.power(10, log_prediction)  # Transform back to the original scale
            st.success(f"The predicted house price is: ${real_prediction:,.2f}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please upload a model to make predictions.")

# Run the app
if __name__ == "__main__":
    prediction_main()