
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from preprocess import get_outlier_index, series_to_supervised, select_feature

def train_test_split(data, test_size=0.2, look_back=1, n_features=None):
    if type(test_size) == int:
        train_size = len(data) - test_size
    else:
        train_size = int(len(data) * (1 - test_size))
        
    train = data[:train_size, :]
    test = data[train_size:, :]

    train_X, train_y = train[:, :-n_features], train[:, -1]
    test_X, test_y = test[:, :-n_features], test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], look_back, n_features))
    test_X = test_X.reshape((test_X.shape[0], look_back, n_features))

    return train_X, train_y, test_X, test_y


def get_data(dataset, look_back=3, test_size=0.2, k_feature=3):
    n_features = len(dataset.columns)
    dataset = dataset.replace('Non', 0)
    dataset = dataset.replace('-', 0)
    data = dataset.values.astype('float32')
    data = MinMaxScaler(feature_range=(0, 1)).fit_transform(data)

    standard_indiecs, outlier_indices = get_outlier_index(data[:, -1].reshape(-1, 1))
    data = data[standard_indiecs, :]

    data = series_to_supervised(data, look_back, 1)
    data = data.values

    origin_train_X, train_y, origin_test_X, test_y = train_test_split(data, test_size=0.2, look_back=look_back, n_features=n_features)
    
    important_feature = select_feature(dataset, origin_train_X, train_y, at_time=-1, k=k_feature)

    train_X, test_X = origin_train_X[:, :, important_feature], origin_test_X[:, :, important_feature]
    print("Detected Ouliers: ")
    for idx in outlier_indices:
        month = (idx % 12) + 1
        year = 1997 + int(idx/12)
        print(f"{month}/{year}")

    return train_X, train_y, test_X, test_y

def plot_forecast_chart(model, test_X, test_y, city_name):
    model.eval()
    y_pred = model.predict(test_X).detach().numpy()
    plt.plot(y_pred, label='predicted')
    plt.plot(test_y, label='actual')
    plt.legend()
    plt.show()

def plot_entire_chart(model, train_X, train_y, test_X, test_y, city_name):
    model.eval()
    true_y = np.hstack((train_y, test_y))

    train_pred = np.empty_like(true_y)
    train_pred[:] = np.nan
    train_pred[:len(train_y)] = model.predict(train_X).detach().numpy()

    test_pred = np.empty_like(true_y)
    test_pred[:] = np.nan
    test_pred[len(train_y):,] = model.predict(test_X).detach().numpy()

    plt.plot(true_y, label="actual")
    plt.plot(train_pred, label='train_predicted')
    plt.plot(test_pred, label='test_predicted')
    plt.xlabel('Month')
    plt.ylabel('Rate')
    plt.title(city_name)
    #plt.savefig('High Quality Image/' + city_name, dpi=2000)
    #plt.savefig('Other/' + city_name, dpi=2000)
    plt.legend()
    plt.show()