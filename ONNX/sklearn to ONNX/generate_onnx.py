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
import inspect
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
import yaml
from onnxmltools import convert_xgboost
from xgboost import XGBClassifier, train, XGBRegressor
import pandas as pd
from src.utils import create_onnx, get_config, generateFiles
import xgboost as xgb


class OnnxGenerator:
    param_dict = {'DecisionTreeClassifier': {'random_state': 0}, 'DecisionTreeRegressor': {'random_state': 0},
                  'RandomForestClassifier': {'random_state': 0}, 'RandomForestRegressor': {'random_state': 0},
                  'XGBRegressor': {'objective': 'reg:squarederror', 'random_state': 42},
                  'XGBClassifier': {'objective': 'reg:squarederror', 'random_state': 42},
                  'XGB': {}}

    def iris(self):
        iris = datasets.load_iris()
        X, y = iris.data, iris.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        mapping_dict = get_config()

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        num_round = 2

        for algo in mapping_dict.get("algo_list"):
            algo_mapping = algo.split("_")
            params = self.param_dict.get(algo_mapping[0])
            global model
            if (algo_mapping[0] == "XGB"):
                params = self.param_dict.get(algo_mapping[0])
                model = xgb.train(params, dtrain, num_round)
                generateFiles(model, ['predict'], dtest, "iris" + algo_mapping[1], '../pred_data')
            elif (params is None):
                model = globals()[algo_mapping[0]]()
                model.fit(X_train, y_train)
                generateFiles(model, ['predict'], X_test, "iris" + algo_mapping[1], '../pred_data')
            else:
                model = globals()[algo_mapping[0]](**params)
                model.fit(X_train, y_train)
                generateFiles(model, ['predict'], X_test, "iris" + algo_mapping[1], '../pred_data')

            create_onnx(model, X, "iris", algo_mapping[1])

    def diabetes(self):
        dataset = datasets.load_diabetes()
        X = dataset.data
        y = dataset.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        mapping_dict = get_config()

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        num_round = 2

        for algo in mapping_dict.get("algo_list"):
            algo_mapping = algo.split("_")
            params = self.param_dict.get(algo_mapping[0])
            global model
            if (algo_mapping[0] == "XGB"):
                params = self.param_dict.get(algo_mapping[0])
                model = xgb.train(params, dtrain, num_round)
                generateFiles(model, ['predict'], dtest, "diabetes" + algo_mapping[1], '../pred_data')

            elif (params is None):
                model = globals()[algo_mapping[0]]()
                model.fit(X_train, y_train)
                generateFiles(model, ['predict'], X_test, "diabetes" + algo_mapping[1], '../pred_data/diabetes')

            else:
                model = globals()[algo_mapping[0]](**params)
                model.fit(X_train, y_train)
                generateFiles(model, ['predict'], X_test, "diabetes" + algo_mapping[1], '../pred_data/diabetes')
            create_onnx(model, X, "diabetes", algo_mapping[1])

    def breast_cancer(self):
        bc = datasets.load_breast_cancer()
        X, y = bc.data, bc.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        mapping_dict = get_config()

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        num_round = 2

        for algo in mapping_dict.get("algo_list"):
            algo_mapping = algo.split("_")
            params = self.param_dict.get(algo_mapping[0])
            global model
            if (algo_mapping[0] == "XGB"):
                params = self.param_dict.get(algo_mapping[0])
                model = xgb.train(params, dtrain, num_round)
                generateFiles(model, ['predict'], dtest, "breast_cancer" + algo_mapping[1], '../pred_data')

            elif (params is None):
                model = globals()[algo_mapping[0]]()
                model.fit(X_train, y_train)
                generateFiles(model, ['predict'], X_test, "breast_cancer" + algo_mapping[1], '../pred_data')

            else:
                if (algo_mapping[0] == "XGBClassifier"):
                    model = globals()[algo_mapping[0]]()
                    model.fit(X_train, y_train)
                    generateFiles(model, ['predict'], X_test, "breast_cancer" + algo_mapping[1], '../pred_data')
                else:
                    model = globals()[algo_mapping[0]](**params)
                    model.fit(X_train, y_train)
                    generateFiles(model, ['predict'], X_test, "breast_cancer" + algo_mapping[1], '../pred_data')

            create_onnx(model, X, "breast_cancer", algo_mapping[1])

    def forestfires(self):
        dataset = pd.read_csv('..\datasets\Forestfires.csv')

        dataset.shape
        dataset = dataset.fillna(method='ffill')

        X = dataset[
            ["X", "Y", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "rain"]].values
        y = dataset["wind"].values
        lb = preprocessing.LabelEncoder()
        y = lb.fit_transform(y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        mapping_dict = get_config()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        num_round = 2

        for algo in mapping_dict.get("algo_list"):
            algo_mapping = algo.split("_")
            params = self.param_dict.get(algo_mapping[0])
            global model
            if (algo_mapping[0] == "XGB"):
                params = self.param_dict.get(algo_mapping[0])
                model = xgb.train(params, dtrain, num_round)
                generateFiles(model, ['predict'], dtest, "forestfires" + algo_mapping[1], '../pred_data')

            elif (params is None):
                model = globals()[algo_mapping[0]]()
                model.fit(X_train, y_train)
                generateFiles(model, ['predict'], X_test, "forestfires" + algo_mapping[1], '../pred_data')

            else:
                model = globals()[algo_mapping[0]](**params)
                model.fit(X_train, y_train)
                generateFiles(model, ['predict'], X_test, "forestfires" + algo_mapping[1], '../pred_data')
            create_onnx(model, X, "forestfires", algo_mapping[1])

    def wine(self):
        dataset = pd.read_csv('..\datasets\winequality.csv')
        dataset.shape
        dataset = dataset.fillna(method='ffill')
        X = dataset[
            ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
             'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']].values
        y = dataset['quality'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        mapping_dict = get_config()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        num_round = 2

        for algo in mapping_dict.get("algo_list"):
            algo_mapping = algo.split("_")
            params = self.param_dict.get(algo_mapping[0])
            global model
            if (algo_mapping[0] == "XGB"):
                params = self.param_dict.get(algo_mapping[0])
                model = xgb.train(params, dtrain, num_round)
                generateFiles(model, ['predict'], dtest, "wine" + algo_mapping[1], '../pred_data')

            elif (params is None):
                model = globals()[algo_mapping[0]]()
                model.fit(X_train, y_train)
                generateFiles(model, ['predict'], X_test, "wine" + algo_mapping[1], '../pred_data')

            else:
                model = globals()[algo_mapping[0]](**params)
                model.fit(X_train, y_train)
                generateFiles(model, ['predict'], X_test, "wine" + algo_mapping[1], '../pred_data')
            create_onnx(model, X, "wine", algo_mapping[1])

    def digits(self):
        dataset = datasets.load_digits()
        X = dataset.data
        y = dataset.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        mapping_dict = get_config()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        num_round = 2

        for algo in mapping_dict.get("algo_list"):
            algo_mapping = algo.split("_")
            params = self.param_dict.get(algo_mapping[0])
            global model
            if (algo_mapping[0] == "XGB"):
                params = self.param_dict.get(algo_mapping[0])
                model = xgb.train(params, dtrain, num_round)
                generateFiles(model, ['predict'], dtest, "digits" + algo_mapping[1], '../pred_data')

            elif (params is None):
                model = globals()[algo_mapping[0]]()
                model.fit(X_train, y_train)
                generateFiles(model, ['predict'], X_test, "digits" + algo_mapping[1], '../pred_data')

            else:
                model = globals()[algo_mapping[0]](**params)
                model.fit(X_train, y_train)
                generateFiles(model, ['predict'], X_test, "digits" + algo_mapping[1], '../pred_data')
            create_onnx(model, X, "digits", algo_mapping[1])

    def mall_customers(self):
        dataset = pd.read_csv('..\datasets\Mall_Customers.csv')

        dataset.shape
        dataset = dataset.fillna(method='ffill')
        number = preprocessing.LabelEncoder()

        dataset['Genre'] = number.fit_transform(dataset['Genre'].astype('str'))

        X = dataset[
            ['CustomerID', 'Genre', 'Age', 'Annual Income']].values
        y = dataset['Spending Score'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        mapping_dict = get_config()
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        num_round = 2

        for algo in mapping_dict.get("algo_list"):
            algo_mapping = algo.split("_")
            params = self.param_dict.get(algo_mapping[0])
            global model
            if (algo_mapping[0] == "XGB"):
                params = self.param_dict.get(algo_mapping[0])
                model = xgb.train(params, dtrain, num_round)
                generateFiles(model, ['predict'], dtest, "mall_customers" + algo_mapping[1], '../pred_data')
            elif (params is None):
                model = globals()[algo_mapping[0]]()
                model.fit(X_train, y_train)
                generateFiles(model, ['predict'], X_test, "mall_customers" + algo_mapping[1], '../pred_data')

            else:
                model = globals()[algo_mapping[0]](**params)
                model.fit(X_train, y_train)
                generateFiles(model, ['predict'], X_test, "mall_customers" + algo_mapping[1], '../pred_data')
            create_onnx(model, X, "mall_customers", algo_mapping[1])


d = OnnxGenerator()
attrs = (getattr(d, name) for name in dir(d))
methods = filter(inspect.ismethod, attrs)
for method in methods:
    method()

