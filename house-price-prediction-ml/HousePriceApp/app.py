import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Page configuration

st.set_page_config(page_title="House Price Predictor", layout="wide")

# Load model

# Load model
with open("house_price_model.pkl", "rb") as file:
    model = pickle.load(file)
# Title

st.title("🏠 House Price Prediction App")
st.write("This ML app predicts the price of a house based on key features.")

# Sidebar inputs

st.sidebar.header("Enter House Details")

GrLivArea = st.sidebar.number_input("Living Area (sq ft)", 300, 10000, 1500)
OverallQual = st.sidebar.slider("Overall Quality", 1, 10, 5)
TotalBsmtSF = st.sidebar.number_input("Basement Area", 0, 5000, 800)
GarageCars = st.sidebar.slider("Garage Capacity", 0, 5, 1)
YearBuilt = st.sidebar.number_input("Year Built", 1900, 2025, 2000)

# Create full feature input (to match model feature count)

feature_count = model.n_features_in_
input_array = np.zeros((1, feature_count))

input_array[0,0] = GrLivArea
input_array[0,1] = OverallQual
input_array[0,2] = TotalBsmtSF
input_array[0,3] = GarageCars
input_array[0,4] = YearBuilt

input_df = pd.DataFrame(input_array)

# Layout columns

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Features")
    st.write({
"GrLivArea": GrLivArea,
"OverallQual": OverallQual,
"TotalBsmtSF": TotalBsmtSF,
"GarageCars": GarageCars,
"YearBuilt": YearBuilt
})

with col2:
    if st.button("Predict Price"):
        prediction = model.predict(input_df)[0]
        st.subheader("Predicted House Price")
        st.success(f"${prediction:,.2f}")
