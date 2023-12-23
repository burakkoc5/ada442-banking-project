import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler

def preprocess_input(user_input):
    # Kullanıcıdan alınan girdiyi modelin beklentisine göre ön işleme
    # Bu kısımda, kullanıcının girdisini modele uygun formata getirin
    # Örneğin: Label encoding, özellik mühendisliği, vb.

    # Seçilen kategorik sütunları label encoding yapalım
    categorical_cols = ['job', 'marital', 'education', 'default', 'contact', 'housing', 'loan', 'month', 'day_of_week', 'poutcome']
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
    exp_transformer = FunctionTransformer(lambda x: x**2, validate=True)
    columns_to_exp = ['nr.employed']
    positively_skewed = exp_transformer.transform(user_input[columns_to_exp])
    user_input['nr.employed'] = positively_skewed[:, 0]

    # Sayısal sütunları standardize edelim
    numeric_cols = ['age', 'duration', 'campaign', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    user_input[numeric_cols] = StandardScaler().fit_transform(user_input[numeric_cols])

    return user_input

def make_prediction(user_input):
    loaded_model = joblib.load('xgboost_banking_model.joblib')
    # Tahmin yapmak için modeli ve kullanıcı girdisini kullanın
    prediction = loaded_model.predict(user_input)
    return prediction

# Streamlit arayüzü
st.title("Bank Marketing Tahmin Uygulaması")

# Kullanıcıdan giriş al
age = st.slider("Yaşınızı Seçin", min_value=18, max_value=100, value=30)
# Diğer girişleri de ekleyebilirsiniz

# Kullanıcının tahmin yapmak için butona tıklamasını sağlayın
if st.button("Tahmin Yap"):
    # Kullanıcının girdisini modele uygun formata getirin
    user_input = pd.DataFrame({
        'age': [age],  # Diğer girdileri de ekleyin
        # 'feature_name': [feature_value],
    })

    # Ön işleme adımlarını uygula
    user_input = preprocess_input(user_input)

    # Tahmin yap
    prediction = make_prediction(user_input)

    # Tahmin sonucunu göster
    st.success(f"Tahmininiz: {prediction}")