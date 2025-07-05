import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

def load_data():
    url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
               'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    df = pd.DataFrame(data, columns=columns)
    df['MEDV'] = target
    return df

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return mse, r2

def get_models(X_train, y_train):
    models = {}

    # Ridge Regression
    ridge_params = {'alpha': [0.1, 1.0, 10.0]}
    ridge = GridSearchCV(Ridge(), ridge_params, cv=5)
    ridge.fit(X_train, y_train)
    models['Ridge'] = ridge

    # Decision Tree
    dt_params = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
    dt = GridSearchCV(DecisionTreeRegressor(), dt_params, cv=5)
    dt.fit(X_train, y_train)
    models['DecisionTree'] = dt

    # Random Forest
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5]
    }
    rf = GridSearchCV(RandomForestRegressor(), rf_params, cv=3)
    rf.fit(X_train, y_train)
    models['RandomForest'] = rf

    return models
