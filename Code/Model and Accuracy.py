#coding=utf-8
import sklearn
import matplotlib.pyplot as plt
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from matplotlib import pyplot
from sklearn.linear_model import LogisticRegression
model_LR= LogisticRegression()
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
model_LR= LogisticRegression()
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

combined_train_test = pd.read_csv('/Users/zhouxueqi/Desktop/feature_selected.csv')
train_data = combined_train_test[:29999]
test_data = combined_train_test[29999:]
print(train_data.shape)
train_data_X = train_data.drop('target',1)
train_data_Y = train_data['target']
test_data_Y = test_data['target']
test_data_X = test_data.drop('target',1)
X = train_data_X
y = train_data_Y

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)


# scaler = StandardScaler() # 标准化转换
# scaler.fit(X_test)  # 训练标准化对象
# x_test_Standard= scaler.transform(X_test)   # 转换数据集
# scaler.fit(X_train)  # 训练标准化对象
# x_train_Standard= scaler.transform(X_train)   # 转换数据集

#
# bp = MLPClassifier(hidden_layer_sizes=(1600,), activation='relu',
#                    solver='lbfgs', alpha=0.0001, batch_size='auto',
#                    learning_rate='constant')
# bp.fit(x_train_Standard, y_train.astype('int'))
# y_pred = bp.predict(x_test_Standard)
#
# y_test1 = y_test.tolist()
# y_pred = list(y_pred)
# print(int(y_test1[1]))
# for i in range(len(y_test1)):
#   y_test1[i] = int(y_test1[i])
# print("actual data：\t",y_test1)
# print("prediction：\t",y_pred)

# get a stacking ensemble of models
def get_stacking():
    # define the base models
    level0 = list()
    level0.append(('lr', LogisticRegression()))
    level0.append(('knn', KNeighborsClassifier()))
    level0.append(('cart', DecisionTreeClassifier()))
    level0.append(('svm', SVC()))
    level0.append(('bayes', GaussianNB()))
    # define meta learner model
    level1 = LogisticRegression()
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    model.fit(X_train, y_train)
    return model

# get a list of models to evaluate
def get_models():
	models = dict()
	models['lr'] = LogisticRegression()
	models['knn'] = KNeighborsClassifier()
	models['cart'] = DecisionTreeClassifier()
	models['svm'] = SVC()
	models['bayes'] = GaussianNB()
	models['stacking'] = get_stacking()
	return models

# get the dataset
def get_dataset():
	X, y = make_classification(n_samples=30000, n_features=200, n_informative=15, n_redundant=5, random_state=1)
	return X, y

# evaluate a give model using cross-validation
def evaluate_model(model):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	return scores

# define dataset
X, y = get_dataset()
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()

pred = models['stacking'].predict(X_test)

y_pred = pd.DataFrame(pred)
y_pred.to_csv('/Users/zhouxueqi/Desktop/predict.csv')
y_test.to_csv('/Users/zhouxueqi/Desktop/y_test.csv')

# # calculate how accurate it is
pred = pd.read_csv('/Users/zhouxueqi/Desktop/predict.csv')
pred.columns = ["ID","prediction"]
y_pred = pred["prediction"]

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr,tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Graph")
plt.show()

auc = roc_auc_score(y_test, y_pred)
print(auc)