import yaml
from skl2onnx.common.data_types import FloatTensorType
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy
from sklearn.linear_model import LogisticRegression, LinearRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as rt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import yaml
import os
from onnxmltools import convert_xgboost
from xgboost import XGBClassifier, train, XGBRegressor
import pandas as pd
import ibserializer.HexEncoder as encoder
import numpy as np
from os import listdir
from os.path import isfile, join
from execute_onnx import fun_execute_onnx


def get_config():
    with open(r'../conf/algo.yaml') as file:
        documents = yaml.full_load(file)
        dict = {}
        for item, doc in documents.items():
            dict[item] = doc

    return dict


def get_X_test(dataset_name):
    if ("diabetes" in dataset_name):

        dataset = datasets.load_diabetes()
        X = dataset.data
        y = dataset.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        return X_test

    elif ("digits" in dataset_name):
        dataset = datasets.load_digits()
        X = dataset.data
        y = dataset.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        return X_test
    elif "wine" in dataset_name:
        dataset = pd.read_csv('..\datasets\winequality.csv')
        dataset.shape
        dataset = dataset.fillna(method='ffill')
        X = dataset[
            ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
             'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']].values
        y = dataset['quality'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        return X_test


def get_lib_pred(file_path):
    list = []
    path = '../pytest_data/pred_data/' + file_path

    f = open(path, 'r')
    lines = f.readlines()
    for line in lines:
        val = float.fromhex(line.rstrip("\n"))
        list.append(float(val))

    return list


def generateTestdataPython():
    list = []
    onnx_models_dir = [f for f in listdir('../pytest_data/onnx_models') if
                       isfile(join('../pytest_data/onnx_models', f))]
    pred_data_dir = [f for f in listdir('../pytest_data/pred_data') if isfile(join('../pytest_data/pred_data', f))]

    global lib_pred_list
    for i in range(len(pred_data_dir)):
        lib_pred_list = get_lib_pred(pred_data_dir[i].__str__())

        X_test = get_X_test(onnx_models_dir[i].__str__())
        if X_test is None:
            print("error for->#################")
            print(onnx_models_dir[i].__str__())
        actual_list = fun_execute_onnx('../pytest_data/onnx_models/' + onnx_models_dir[i].__str__(), X_test)

        for k in range(len(lib_pred_list)):
            tup = (lib_pred_list[k], actual_list[k], onnx_models_dir[i].__str__())
            list.append(tup)

    return list


def generateFiles(estimator, funcs, X_test, testset_name, path):
    # cpath = path + '/' + testset_name
    # os.makedirs(cpath)
    for func in funcs:
        lib_preds = getattr(estimator, func)(X_test).tolist()
        encode_pred = encoder.encode(lib_preds)
        # with open(cpath + '/library_' + func + '_' + testset_name + '.csv', 'wb') as f:
        # csv.writer(f).writerows(encode_pred)
        np.savetxt(path + '/library_' + func + '_' + testset_name + '.csv', encode_pred, fmt='%s', delimiter=",")

    return


def create_onnx(model, X, dataset, algo):
    algo_name = dataset + algo
    initial_type = [('input', FloatTensorType([None, X.shape[1]]))]
    if ("xgb" in algo_name):
        onx = convert_xgboost(model, 'tree-based classifier',
                              [('input', FloatTensorType([None, X.shape[1]]))])
        with open(r'../onnx_models/{}/{}.onnx'.format(dataset, algo_name), "wb") as f:
            f.write(onx.SerializeToString())
    else:

        onx = convert_sklearn(model=model, initial_types=initial_type)
        with open(r'../onnx_models/{}/{}.onnx'.format(dataset, algo_name), "wb") as f:
            f.write(onx.SerializeToString())
