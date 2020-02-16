import pandas as pd
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def read_description(path_description='../Data/data_description.txt'):
    cols_description = {}
    col, i, j = 'feature_name', 0, 0
    with open(path_description) as file:
        line = file.readline()
        while len(line) > 0:
            if line[0].isalpha():
                col = line.split(':')[0]
                cols_description[col] = {}
                j = 0
            elif len(line) >= 10:
                keys = line.split()
                if (col == 'MSZoning' and keys[0].lower() == 'c') or (col == 'Exterior1st' and keys[0].lower() == 'wd') \
                        or (col == 'Exterior2nd' and (keys[0].lower() == 'wd' or keys[0].lower() == 'brk')):
                    key = keys[0] + ' ' + keys[1]
                else:
                    key = keys[0]
                cols_description[col][key.lower()] = j
                j += 1
            line = file.readline()
    return cols_description


def process_data(X=pd.DataFrame(), path_description='../Data/data_description.txt'):
    cols_description = read_description(path_description)
    cols = []
    data = copy(X)
    # set value for column is string
    for col in list(data.columns.values):
        if data.dtypes[col] == np.object:
            data[col] = data[col].str.lower().replace(to_replace=cols_description[col].keys(),
                                                      value=cols_description[col].values())
        else:
            cols.append(col)
    # fill cell NaN
    data.fillna(data.median(), inplace=True)
    # normalize data and discrete
    data_number_norm = MinMaxScaler().fit_transform(data[cols].to_numpy())

    data_number_norm = pd.DataFrame(data_number_norm * 10, columns=cols,
                                    index=data.index, dtype=int).round()
    data[cols] = data_number_norm

    return data, cols, data.index


def predict_data(model, X=None, path_file_csv='model.csv', index=None):
    X_test = copy(X)
    y = model.predict(X_test)
    result = pd.DataFrame({'Id': index, 'SalePrice': y})
    result.to_csv(path_file_csv, index=False)


if __name__ == '__main__':
    path_train = '../Data/train.csv'
    path_test = '../Data/test.csv'
    # load data, process data
    X_train = pd.read_csv(path_train, index_col='Id')
    features = list(X_train.columns.values)
    y_train = X_train[features[-1]]
    del X_train[features[-1]]
    X_train, col_number, index_train = process_data(X_train)

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, train_size=0.8, random_state=42)
    print('Features name :', features[:-2])
    print(read_description())

    # Not Scaling, RandomForest and GradientBoosting

    # RandomForest
    random_tree = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42, oob_score=True)
    param_grid = [{'n_estimators': np.linspace(50, 500, 10, dtype=int).tolist()}]
    gs_randomforest = GridSearchCV(random_tree, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error',
                                   n_jobs=-1)

    gs_randomforest.fit(X_train, y_train)

    # print results of GridSearch
    for mean, param in zip(gs_randomforest.cv_results_['mean_test_score'], gs_randomforest.cv_results_['params']):
        print(np.sqrt(-mean), ':', param)
    print('Best params ', gs_randomforest.best_params_)
    random_tree.set_params(n_estimators=gs_randomforest.best_params_['n_estimators'])
    random_tree.fit(X_train, y_train)

    y_test_predict = random_tree.predict(X_test)
    print('RandomForest MSE train : ', np.sqrt(mean_squared_error(y_test, y_test_predict)))

    # ----------------------------------------------------------------------------

    gb = GradientBoostingRegressor(n_estimators=500, random_state=42, learning_rate=0.3)

    param_grid = [
        {'n_estimators': np.linspace(50, 500, 10, dtype=int).tolist(),
         'learning_rate': np.linspace(0.1, 0.9, 9).tolist()}]
    gs_gb = GridSearchCV(gb, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    gs_gb.fit(X_train, y_train)

    for mean, param in zip(gs_gb.cv_results_['mean_test_score'], gs_gb.cv_results_['params']):
        print(np.sqrt(-mean), ':', param)
    print('Best params ', gs_gb.best_params_)

    gb.set_params(n_estimators=gs_gb.best_params_['n_estimators'], learning_rate=gs_gb.best_params_['learning_rate'])
    gb.fit(X_train, y_train)

    y_test_predict_gb = gb.predict(X_test)

    print('GradientBoosting MSE train : ', np.sqrt(mean_squared_error(y_test, y_test_predict_gb)))

    # predict data from test.csv
    X_test_real = pd.read_csv(path_test, index_col='Id')
    X_test_real, _, index_test = process_data(X_test)
    print('RandomForest Predict test.csv')
    predict_data(random_tree, X_test_real, '../Data/test_random_tree.csv', index_test)
    print('GradientBoosting Predict test.csv')
    predict_data(gb, X_test_real, '../Data/test_gradient_boosting.csv', index_test)
    # # plot
    # x = [i for i in range(1, X_train.shape[0] + 1)]
    # plt.figure(figsize=(10, 4))
    #
    # plt.subplot(1, 3, 1)
    # plt.scatter(x, y_train, s=y_train / 10000, c=y_train / 10000, cmap='jet')
    # plt.ylim([0, 1000000])
    # plt.title('Original')
    #
    # plt.subplot(1, 3, 2)
    # plt.scatter(x, y_train_predict, s=y_train_predict / 10000, c=y_train_predict / 10000, cmap='jet')
    # plt.ylim([0, 1000000])
    # plt.title('RandomForest')
    #
    # plt.subplot(1, 3, 3)
    # plt.scatter(x, y_train_predict_gb, s=y_train_predict_gb / 10000, c=y_train_predict_gb / 10000, cmap='jet')
    # plt.ylim([0, 1000000])
    # plt.title('GradientBoosting')

    # plt.show()
