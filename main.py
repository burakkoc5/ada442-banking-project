import pickle

import streamlit as st
import numpy as np
import pandas as pd  # To read the file
import warnings

warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder# Label Encoding process (In the preprocessing part)
from sklearn.preprocessing import FunctionTransformer  # Transformation process (In the preprocessing part)
from sklearn.preprocessing import StandardScaler

def preprocess_input(user_input):
    # Kullanıcıdan alınan girdiyi modelin beklentisine göre ön işleme
    # Bu kısımda, kullanıcının girdisini modele uygun formata getirin
    # Örneğin: Label encoding, özellik mühendisliği, vb.

    # Seçilen kategorik sütunları label encoding yapalım
    categorical_cols = ['job', 'marital', 'education', 'default', 'contact', 'housing', 'loan', 'month', 'day_of_week',
                        'poutcome']
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        user_input[col] = label_encoder.fit_transform(user_input[col])

    # Negatif skewness olan sütunlarda log transform yapalım
    log_transformer = FunctionTransformer(np.log1p, validate=True)
    columns_to_log = ['age', 'duration', 'campaign', 'previous']
    negatively_skewed = log_transformer.transform(user_input[columns_to_log])
    user_input['age'] = negatively_skewed[:, 0]
    user_input['duration'] = negatively_skewed[:, 1]
    user_input['campaign'] = negatively_skewed[:, 2]
    user_input['previous'] = negatively_skewed[:, 3]

    # Pozitif skewness olan sütunlarda x^2 transform yapalım
    exp_transformer = FunctionTransformer(lambda x: x ** 2, validate=True)
    columns_to_exp = ['nr.employed']
    positively_skewed = exp_transformer.transform(user_input[columns_to_exp])
    user_input['nr.employed'] = positively_skewed[:, 0]

    # Sayısal sütunları standardize edelim
    numeric_cols = ['age', 'duration', 'campaign', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                    'euribor3m', 'nr.employed']
    user_input[numeric_cols] = StandardScaler().fit_transform(user_input[numeric_cols])

    return user_input


def make_prediction(user_input):
    pickled_model = pickle.load(open('model.pkl', 'rb'))

    # Tahmin yapmak için modeli ve kullanıcı girdisini kullanın
    prediction = pickled_model.predict(user_input)
    return prediction

if __name__ == '__main__':
    st.title("Bank Marketing Prediction App")
    job = st.selectbox("Job",["blue-collar", "services", "admin.", "entrepreneur", "self-employed", "technician",
         "management", "student", "retired", "housemaid", "unemployed"])
    marital_status = st.selectbox("Marital Status", ["married", "single", "divorced", "unknown"])
    education = st.selectbox("Education",
                             ["basic.9y", "high.school", "university.degree", "professional.course", "basic.6y",
                              "basic.4y", "illiterate", "unknown"])
    default = st.selectbox("Default", ["no", "yes", "unknown"])
    housing = st.selectbox("Housing", ["no", "yes", "unknown"])
    loan = st.selectbox("Loan", ["no", "yes", "unknown"])
    contact = st.selectbox("Contact", ["cellular", "telephone", "unknown"])
    month = st.selectbox("Month", ["may", "jun", "nov", "sep", "jul", "aug", "mar", "oct", "apr", "dec"])
    day_of_week = st.selectbox("Day of week", ["fri","wed","mon","thu","tue"])
    age = st.slider("Age", min_value=0.0, max_value=88.0, value=25.0)
    duration = st.slider("Duration", min_value=0.0, max_value=3643.0, value=300.0)
    campaign = st.slider("Campaign", min_value=0.0, max_value=35.0, value=10.0)
    pdays = st.slider("Pdays", min_value=0.0, max_value=999.0, value=15.0)
    previous = st.slider("Previous", min_value=0.0, max_value=6.0, value=5.0)
    poutcome = st.selectbox("Poutcome", ["nonexistent", "failure", "success"])
    emp_var_rate = st.slider("Employment Variation Rate", min_value=-3.0, max_value=1.0, value=0.0)
    cons_price_idx = st.slider("Consumer Price Index", min_value=0.0, max_value=94.0, value=35.0)
    cons_conf_idx = st.slider("Consumer Confidence Index", min_value=-50.0, max_value=0.0, value=-35.0)
    euribor3m = st.slider("Euribor 3 Month Rate", min_value=0.0, max_value=5.0, value=3.0)
    nr_employed = st.slider("Number of Employees", min_value=0.0, max_value=5228.0, value=5000.0)

    # Create a dataframe with the user input
    input_data = pd.DataFrame({
        'age': [age],
        'job': [job],
        'marital': [marital_status],
        'education': [education],
        'default': [default],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'month': [month],
        'day_of_week': [day_of_week],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'poutcome': [poutcome],
        'emp.var.rate': [emp_var_rate],
        'cons.price.idx': [cons_price_idx],
        'cons.conf.idx': [cons_conf_idx],
        'euribor3m': [euribor3m],
        'nr.employed': [nr_employed],
    })

    # Preprocess the input data
    input_data = preprocess_input(input_data)

    # Display the preprocessed input data
    st.subheader("Preprocessed Input Data")
    st.write(input_data)

    # Load the trained model

    # Make predictions
    prediction = make_prediction(input_data)

    # Display the prediction
    st.subheader("Prediction")
    st.write(prediction)