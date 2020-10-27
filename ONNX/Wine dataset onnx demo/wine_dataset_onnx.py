import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from json.decoder import NaN
from onnxmltools import convert_xgboost
from skl2onnx.common.data_types import FloatTensorType
import xgboost as xgb
import onnxruntime as rt
import numpy

dataset = pd.read_csv('winequality.csv')
dataset.shape
dataset = dataset.fillna(method='ffill')
X = dataset[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide',
             'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']].values
y = dataset['quality'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

estimator = LogisticRegression(random_state=0)
estimator.fit(X_train, y_train)

dtrain = xgb.DMatrix(X_train, label=y_train)
model = xgb.train({'objective': 'multi:softprob', 'num_class': 10}, dtrain)

onx = convert_xgboost(model, 'tree-based classifier',
                      [('input', FloatTensorType([None, X.shape[1]]))])

with open(r'myxgb.onnx', "wb") as f:
    f.write(onx.SerializeToString())

sess = rt.InferenceSession('myxgb.onnx')

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name

pred_onx = sess.run([label_name], {input_name: X.astype(numpy.float32)})[0]

print(np.unique(pred_onx))
