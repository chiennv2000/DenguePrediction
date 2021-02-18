import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor

def preprocess_data(data, city, disease, mode='rate') -> pd.DataFrame:
    city_data = data.loc[data['province'].isin([city])]
    city_data.drop(columns=['year_month', 'province'], inplace=True)
    if mode == 'full':
        return city_data
    if disease == 'Influenza' and mode == 'rate':
        sub_data = city_data.drop(columns=['Influenza_cases', 'Dengue_fever_cases', 'Diarrhoea_cases', 'Dengue_fever_rates', 'Diarrhoea_rates'], axis=1, inplace=False)
    elif disease == 'Dengue_fever' and mode == 'rate':
        sub_data = city_data.drop(columns=['Influenza_cases', 'Dengue_fever_cases', 'Diarrhoea_cases', 'Influenza_rates', 'Diarrhoea_rates'], axis=1, inplace=False)
    elif disease == 'Diarrhoea' and mode == 'rate':
        sub_data = city_data.drop(columns=['Influenza_cases', 'Dengue_fever_cases', 'Diarrhoea_cases', 'Influenza_rates', 'Dengue_fever_rates'], axis=1, inplace=False)
    return sub_data

def series_to_supervised(data, n_in, n_out, dropna=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = [], []

    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    seq_data = pd.concat(cols, axis=1)
    seq_data.columns = names

    if dropna:
        seq_data.dropna(inplace=True)
    return seq_data

def get_outlier_index(otl_X):
    #iso = IsolationForest(contamination=0.01)
    ##ee = EllipticEnvelope()
    lof = LocalOutlierFactor(n_neighbors=15)
    outlier_y = lof.fit_predict(otl_X)
    return np.where(outlier_y == 1)[0], np.where(outlier_y == -1)[0]

def select_feature(dataset, train_X, train_y, at_time=-1, k=10):
    rfe = RFE(RandomForestRegressor(n_estimators=500, random_state=1), n_features_to_select=k)
    fit = rfe.fit(train_X[:, at_time, :], train_y)
    important_feature = []
    print("Important Feature:")
    for i in range(len(fit.support_)):
        if fit.support_[i]:
            important_feature.append(i)
            print(' - %s' % (dataset.columns[i]))
    return np.array(important_feature)


