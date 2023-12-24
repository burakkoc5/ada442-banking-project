import numpy as np
import pandas as pd  # To read the file
import warnings
import pickle

warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder# Label Encoding process (In the preprocessing part)
from sklearn.preprocessing import FunctionTransformer  # Transformation process (In the preprocessing part)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import \
    cross_val_score  # Computes scores through cross-validation for model performance evaluation.
from sklearn.model_selection import \
    train_test_split  # Splits a dataset into training and testing subsets for model assessment.
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE


csv_file_path = "bank-additional.csv"  # Local path of the .csv file

data = pd.read_csv(csv_file_path, sep=';')  # Loading the data set into the "data" variable by seperating w.r.t ';'


# Finding unique values for each column and storing in the variable "unique_values"
unique_values = data.nunique()



categorical_cols = ['job', 'marital', 'education', 'default', 'contact', 'housing', 'loan', 'month', 'day_of_week',
                    'poutcome', "y"]
numeric_cols = [col for col in data.columns if col not in categorical_cols]

print("Categorical Columns: " + str(categorical_cols))
print("\nNumeric Columns: " + str(numeric_cols))


### !!!!!
numeric_cols.remove("previous")  # Causes error on correlation matrix?

for column in numeric_cols:
    # Calculating quartiles and interquartile range
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    upper_limit = Q3 + 1.5 * IQR  # Determining the upper limit

    data.loc[data[column] > upper_limit, column] = upper_limit  # Replace outliers with the upper limit


for col in categorical_cols:
    data[col] = LabelEncoder().fit_transform(data[col])

data.head()  # Printing the data to observe the change (For example, for month column, may got the value 6 and jun got the value 4)

logarithm_transformer = FunctionTransformer(np.log1p,
                                            validate=True)  # Log transform instance from the Function Transformer
exp_transformer = FunctionTransformer(lambda x: x ** 2,
                                      validate=True)  # x² transform instance from the Function Transformer (Manually)

# Applying log transforms for the negatively skewed columns.
columns = ['age', 'duration', 'campaign', 'previous']  # Determined by analyzing the distributions of the columns

negatively_skewed = logarithm_transformer.transform(data[columns])
data['age'] = negatively_skewed[:, 0]
data['duration'] = negatively_skewed[:, 1]
data['campaign'] = negatively_skewed[:, 2]
data['previous'] = negatively_skewed[:, 3]

# Applying x² transforms for the negatively skewed columns.
columns = ['nr.employed']  # Determined by analyzing the distributions of the columns

positively_skewed = exp_transformer.transform(data[columns])
data['nr.employed'] = positively_skewed[:, 0]


data[numeric_cols] = StandardScaler().fit_transform(
    data[numeric_cols])  # Automatically scaling the numeric columns with Standard Scaler
data.head()  # Displaying the data to visualize the change (For example: campaign, pdays, emp.var.rate, euribor3m)


X = data.drop(["y"], axis=1)
y = data["y"]



# Splitting the dataset into training and testing sets.
X_trainv1, X_test, y_trainv1, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42)  # test_size = 0.3 means 30% of the data will be allocated to the test set.
# Handling the imbalance in the dataset by oversampling the "yes" samples using SMOTE.
# When there are fewer "yes" samples, the model tends to predict the result as "no" more often,
# resulting in a lower recall rate. SMOTE increases the number of "yes" samples to improve the recall rate.
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_trainv1, y_trainv1)


model = XGBClassifier()

model.fit(X_train, y_train)
y_pred_xgb = model.predict(X_test)

param_grid = {"max_depth": range(3, 10)}

grid = GridSearchCV(XGBClassifier(), param_grid, cv=5)
grid.fit(X_train, y_train)

print(grid.best_params_)

from xgboost import XGBClassifier

model = XGBClassifier(max_depth=7)

model.fit(X_train, y_train)
y_pred_xgb = model.predict(X_test)


pickle.dump(model, open('model.pkl', 'wb'))

