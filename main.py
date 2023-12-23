import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler


# -*- coding: utf-8 -*-
"""Banking_Classification (2) (1).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18KDvOCsF28iH_GaYhpC5HP8tW5h6T3DK

# Bank Marketing Classification

## Imports
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd # To read the file
import seaborn as sns
import matplotlib.pyplot as plt
# Enabling inline plotting for Matplotlib in Jupyter Notebook (To display graphs in cells)
# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import zscore # To handle outliers (In the preprocessing part)
from sklearn.preprocessing import LabelEncoder # Label Encoding process (In the preprocessing part)
from sklearn.preprocessing import FunctionTransformer # Transformation process (In the preprocessing part)
from sklearn.preprocessing import StandardScaler # Feature Scaling process (In the preprocessing part)

"""## Loading the dataset"""

csv_file_path ="bank-additional.csv" # Local path of the .csv file

data = pd.read_csv(csv_file_path, sep=';') # Loading the data set into the "data" variable by seperating w.r.t ';'

data.head() # Displaying the first 5 rows of the file to check if the file loaded without an error

"""## Analyzing the dataset before the preprocessing"""

# Finding unique values for each column and storing in the variable "unique_values"
unique_values = data.nunique()

# Displaying the unique values to examine dataset
unique_values

# Checking for null values and their counts
data.isna().sum()

# Checking for duplicate values
data.duplicated().any()

"""### Analyzing some of the columns' distribution (To determine if they need transformations)"""

sns.histplot(x = data["nr.employed"])
plt.title("nr.employed Distribution")
plt.show()

# We can clearly see that "nr.employed" column has positively skewed distribution. (Concentration on larger values)

sns.histplot(x = data["age"])
plt.title("age Distribution")
plt.show()

# We can clearly see that "age" column has negatively skewed distribution. (Concentration on smaller values)

sns.histplot(x = data["duration"])
plt.title("duration Distribution")
plt.show()

# We can clearly see that "duration" column has negatively skewed distribution. (Concentration on smaller values)

sns.countplot(x = data["campaign"])
plt.title("campaign Distribution")
plt.show()

# We can clearly see that "campaign" column has negatively skewed distribution. (Concentration on smaller values)

sns.histplot(x = data["previous"])
plt.title("previous Distribution")
plt.show()

# We can clearly see that "previous" column has negatively skewed distribution. (Concentration on smaller values)

"""## Preprocessing the dataset

### Seperating the categorical and numeric columns into the variables "categorical_cols" and "numeric_cols"
"""

categorical_cols = ['job', 'marital', 'education', 'default','contact','housing', 'loan', 'month', 'day_of_week', 'poutcome', "y"]
numeric_cols = [col for col in data.columns if col not in categorical_cols]

print("Categorical Columns: " + str(categorical_cols))
print("\nNumeric Columns: " + str(numeric_cols))

"""### Finding the outliers on the dataset by Interquartile Method and changing them to upper limit value for that column"""

### !!!!!
numeric_cols.remove("previous") # Causes error on correlation matrix?

for column in numeric_cols:
    # Calculating quartiles and interquartile range
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    upper_limit = Q3 + 1.5 * IQR # Determining the upper limit

    data.loc[data[column] > upper_limit, column] = upper_limit # Replace outliers with the upper limit


# z score method (Not in use because it changes all of the values no matter the threshold value. There may be an implementation mistake.)
"""
z_score_treshold = 2 # Generally chosen as 2 (Same outlier count found for the threshold 3)

for column in numeric_cols:
    z_scores = zscore(data[column]) # Calculating the z score for the column
    outliers = (z_scores > z_score_treshold) | (z_scores < z_score_treshold) # Determining outliers by comparing with the threshold
    column_outlier_count = outliers.sum() # "outliers" is boolen array, so we count the "True" values in the array
    print("Outlier count for the column " + column + " is " + str(column_outlier_count) + ".")
    data.loc[outliers, column] = data[column].median() # If an outlier found, change it by the median of that column
"""

"""### Label Encoding for categorical values"""

for col in categorical_cols:
    data[col] = LabelEncoder().fit_transform(data[col])

data.head() # Printing the data to observe the change (For example, for month column, may got the value 6 and jun got the value 4)

"""### Transformation w.r.t analysis on the distributions."""

logarithm_transformer = FunctionTransformer(np.log1p, validate=True) # Log transform instance from the Function Transformer
exp_transformer = FunctionTransformer(lambda x:x**2, validate=True) # x² transform instance from the Function Transformer (Manually)

#Applying log transforms for the negatively skewed columns.
columns = ['age', 'duration', 'campaign', 'previous'] # Determined by analyzing the distributions of the columns

negatively_skewed = logarithm_transformer.transform(data[columns])
data['age'] = negatively_skewed[:, 0]
data['duration'] = negatively_skewed[:, 1]
data['campaign'] = negatively_skewed[:, 2]
data['previous'] = negatively_skewed[:, 3]

#Applying x² transforms for the negatively skewed columns.
columns = ['nr.employed']# Determined by analyzing the distributions of the columns

positively_skewed = exp_transformer.transform(data[columns])
data['nr.employed'] = positively_skewed[:, 0]

"""### Feature Scaling"""

data[numeric_cols] = StandardScaler().fit_transform(data[numeric_cols]) # Automatically scaling the numeric columns with Standard Scaler
data.head() # Displaying the data to visualize the change (For example: campaign, pdays, emp.var.rate, euribor3m)

"""## Correlation Matrix"""

#İleride buradan bir çıkarım yapılmıyorsa silinebilir
corr = data.corr()
plt.figure(figsize=(14,7))
sns.heatmap(corr,annot=True)
plt.show()

"""## Seperating Input/Output"""

X = data.drop(["y"], axis = 1)
y = data["y"]

"""## Model Selection

### Some Steps
"""
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score # Computes scores through cross-validation for model performance evaluation.
from sklearn.model_selection import train_test_split # Splits a dataset into training and testing subsets for model assessment.

# Splitting the dataset into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # test_size = 0.3 means 30% of the data will be allocated to the test set.

"""### XGBoost"""

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

model = XGBClassifier()

model.fit(X_train, y_train)
y_pred_xgb = model.predict(X_test)

print("Training score:", np.mean(cross_val_score(model, X_train, y_train)))
print("Test score:", np.mean(cross_val_score(model, X_test, y_test)))
print( f"Model Score: {model.score(X_test, y_test)}")
print("\nAccuracy:", accuracy_score(y_test, y_pred_xgb))

param_grid = {"max_depth":range(3,10) }

grid = GridSearchCV(XGBClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

print(grid.best_params_)


from xgboost import XGBClassifier

model = XGBClassifier(max_depth = 3)

model.fit(X_train, y_train)
y_pred_xgb = model.predict(X_test)

print("Training score:", np.mean(cross_val_score(model, X_train, y_train)))
print("Test score:", np.mean(cross_val_score(model, X_test, y_test)))
print( f"Model Score: {model.score(X_test, y_test)}")
print("\nAccuracy:", accuracy_score(y_test, y_pred_xgb))

"""### Confusion Matrix"""

from sklearn.metrics import confusion_matrix

trueNegative, falsePositive, falseNegative, truePositive = confusion_matrix(y_test, y_pred_xgb).ravel()

print("TP: " + str(truePositive))
print("TN: " + str(trueNegative))
print("FP: " + str(falsePositive))
print("FN: " + str(falseNegative))
print("")

recall = truePositive / (truePositive + falseNegative) * 100
precision = truePositive / (truePositive + falsePositive) * 100

print("Recall: " + str(recall) + " %")
print("Precision: " + str(precision) + " %")

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_xgb))



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
    # Tahmin yapmak için modeli ve kullanıcı girdisini kullanın
    prediction = model.predict(user_input)
    return prediction

# Streamlit arayüzü
st.title("Bank Marketing Tahmin")

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