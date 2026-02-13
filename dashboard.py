import streamlit as st
import pandas as pd
import joblib

st.title("Customer Churn Prediction Dashboard")

df = pd.read_csv("telco_churn_featured.csv")
model = joblib.load("churn_model.pkl")

st.subheader("Dataset Preview")
st.dataframe(df.head(40))

st.subheader("Churn Rate")
churn_rate = df['Churn'].mean() * 100
st.metric("Churn Percentage", f"{churn_rate:.2f}%")

st.subheader("Predict Customer Churn")

input_data = df.drop('Churn', axis=1).iloc[0:1]
prediction = model.predict(input_data)

st.write("Prediction:", "Churn" if prediction[0] == 1 else "Not Churn")
