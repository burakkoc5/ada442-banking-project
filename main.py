import pickle

import joblib
import streamlit as st
import numpy as np
import pandas as pd  # To read the file
import warnings

warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder  # Label Encoding process (In the preprocessing part)
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
def make_prediction(model, user_input):
    # Tahmin yapmak için modeli ve kullanıcı girdisini kullanın
    model_prediction = model.predict(user_input)

    print('Model Prediction is ' + str(model_prediction))
    if model_prediction == 0:
        return 'No'
    else:
        return 'Yes'


def button_onclick(input_model, user_data):
    age_text_input = user_data['age']
    duration_text_input = user_data['duration']
    pdays_text_input = user_data['pdays']
    previous_text_input = user_data['previous']

    for column in input_data.columns:
        for index, value in input_data[column].items():
            if pd.isna(value) or value == '':
                st.error("Please enter valid values for empty fields")
                st.stop()

    user_data['duration'] = float(pdays_text_input)
    user_data['previous'] = float(previous_text_input)
    user_data['age'] = float(age_text_input)
    user_data['duration'] = float(duration_text_input)

    # Butona basıldığında preprocess ve prediction metotlarını çağır
    processed_data = preprocess_input(user_data)
    result = make_prediction(input_model, processed_data)

    # Display the prediction
    st.subheader("Prediction")
    if result == 'Yes':
        st.write(f"<span style='font-size:20px; color:green'>{result}</span>", unsafe_allow_html=True)
    else:
        st.write(f"<span style='font-size:20px; color:red'>{result}</span>", unsafe_allow_html=True)


if __name__ == '__main__':
    st.title("Bank Marketing Prediction App")

    # Genişlik değerlerini uygun bir şekilde değiştirin
    width = 300

    age = st.text_input("age", width=width, help='Age of the client')
    job = st.selectbox("job", ["blue-collar", "services", "admin.", "entrepreneur", "self-employed", "technician",
                               "management", "student", "retired", "housemaid", "unemployed"], width=width,
                       help='Type of job (categorical: "admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed")')
    marital_status = st.selectbox("marital", ["married", "single", "divorced", "unknown"], width=width,
                                  help='Marital status (categorical: "divorced", "married", "single", "unknown")')
    education = st.selectbox("education",
                             ["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
                              "professional.course", "university.degree", "unknown"], width=width,
                             help='Education level (categorical: "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree", "unknown")')
    default = st.selectbox("default", ["no", "yes", "unknown"], width=width, help='Has credit in default?')
    balance = st.text_input("balance", width=width, help='Average yearly balance in euros')
    housing = st.selectbox("housing", ["no", "yes", "unknown"], width=width, help='Has housing loan?')
    loan = st.selectbox("loan", ["no", "yes", "unknown"], width=width, help='Has personal loan?')
    contact = st.selectbox("contact", ["cellular", "telephone", "unknown"], width=width,
                           help='Contact communication type (categorical: "cellular", "telephone", "unknown")')
    day_of_week = st.selectbox("day_of_week", ["mon", "tue", "wed", "thu", "fri"], width=width,
                               help='Last contact day of the week')
    month = st.selectbox("month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
                         width=width, help='Last contact month of the year')
    duration = st.text_input('duration', width=width,
                             help='Last contact duration in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.')
    campaign = st.text_input("campaign", width=width,
                             help='Number of contacts performed during this campaign and for this client (numeric, includes last contact)')
    pdays = st.text_input("pdays", width=width,
                          help='Number of days that passed by after the client was last contacted from a previous campaign (numeric; -1 means client was not previously contacted)')
    previous = st.text_input("previous", width=width,
                             help='Number of contacts performed before this campaign and for this client')
    poutcome = st.selectbox("poutcome", ["failure", "nonexistent", "success"], width=width,
                            help='Outcome of the previous marketing campaign')

    emp_var_rate = st.slider("emp_var_rate", min_value=-4.0, max_value=2.0, value=0.0, help='Employment variation rate')
    cons_price_idx = st.slider("cons_price_idx", min_value=0.0, max_value=100.0, value=30.0,
                               help='Consumer price index')
    cons_conf_idx = st.slider("cons_conf_idx", min_value=-70.0, max_value=0.0, value=-20.0,
                              help='Consumer confidence index')
    euribor3m = st.slider("euribor3m", min_value=0.0, max_value=8.0, value=3.0, help='Euribor 3 month rate')
    nr_employed = st.slider("nr_employed", min_value=0.0, max_value=6000.0, value=2000.0, help='Number of employees')

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

    # Load the trained model
    model = joblib.load("model.pkl")

    # Butona basılmadan preprocess ve prediction metodlarını çağırma
    if st.button('Make Prediction'):
        button_onclick(model, input_data)
