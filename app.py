
import streamlit as st
import pandas as pd
import pickle

st.title("アヤメの品種分類アプリ")

with open("model_iris.pkl", "rb") as f:
    model = pickle.load(f)

sepal_length = st.slider("がくの長さ (cm)", 4.0, 8.0, 5.8)
sepal_width = st.slider("がくの幅 (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("花びらの長さ (cm)", 1.0, 7.0, 4.35)
petal_width = st.slider("花びらの幅 (cm)", 0.1, 2.5, 1.3)

input_data = pd.DataFrame(
    [[sepal_length, sepal_width, petal_length, petal_width]],
    columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
)

if st.button("分類する"):
    prediction = model.predict(input_data)[0]
    iris_names = ["setosa（セトサ）", "versicolor（バーシカラー）", "virginica（バージニカ）"]
    st.success(f"予測された品種: {iris_names[prediction]}")
