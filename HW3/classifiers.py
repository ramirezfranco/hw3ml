'''
Auxiliary functions to do classifiers
Assignment 3 
CAPP 30254  Machine Learning for Public Policy

Author: Jesus I. Ramirez Franco
''' 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy 
from sklearn.metrics import f1_score as f1
from sklearn.metrics import precision_score as precision 
from sklearn.metrics import recall_score as recall
from sklearn.metrics import roc_auc_score as roc_auc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import clean
import math 


def temporal_validation_split(df, boundaries_list, outcome_var, date_var):
	df[date_var] = pd.to_datetime(df[date_var])
	sets_list = []
	for i in range(1, len(boundaries_list)-1):
		start = i
		end = i + 1
		train = clean.between_timestamp(df, date_var, boundaries_list[start], boundaries_list[0])
		test = clean.between_timestamp(df, date_var, boundaries_list[end], boundaries_list[start])
		test = test.drop(date_var, axis = 1)
		train = train.drop(date_var, axis = 1)
		sets_list.append(train[outcome_var])
		sets_list.append(train.drop(outcome_var, axis = 1))
		sets_list.append(test[outcome_var])
		sets_list.append(test.drop([outcome_var], axis = 1))
	return sets_list


def precision_top(y_pred, y_test, top):
	max_size = math.ceil((len(y_pred) * top))
	y_pred_t = y_pred[:max_size]
	y_test_t = y_test[:max_size]
	return precision(y_pred_t, y_test_t)


def recall_top(y_pred, y_test, top):
	max_size = math.ceil((len(y_pred) * top))
	y_pred_t = y_pred[:max_size]
	y_test_t = y_test[:max_size]
	return recall(y_pred_t, y_test_t)


def knn_pred(data_set, x_train, y_train, k, weight_type):
    knn = KNeighborsClassifier(n_neighbors=k, weights = weight_type)
    knn.fit(x_train, y_train)
    return knn.predict(data_set)


def knn_dif_models(k_list, weights_list, results_dict, x_train, y_train, x_test):
    for k in k_list:
        for weight in weights_list:
            params = 'knn: ' + str(k) + ', ' + str(weight)
            y_pred = knn_pred(x_test, x_train, y_train, k, weight)
            results_dict[params] = y_pred


def dtree_pred(data_set, x_train, y_train, criterion, splitter, depth):
	dec_tree = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=depth)
	dec_tree.fit(x_train, y_train)
	return dec_tree.predict(data_set)


def dtree_dif_models(results_dict, criterion_list, splitter_list, max_depth_list, x_train, y_train, x_test):
    results_list = []
    for criterion in criterion_list:
        for splitter in splitter_list:
            for depth in max_depth_list:
                params = 'd_tree: ' + str(criterion) + ', ' + str(splitter) + ', ' + str(depth)
                y_pred = dtree_pred(x_test, x_train, y_train, criterion, splitter, depth)
                results_dict[params] = y_pred


def logistic_pred(data_set, x_train, y_train, c, penalty):
	logireg = LogisticRegression(C=c, penalty=penalty)
	logireg.fit(x_train, y_train)
	return logireg.predict(data_set)


def logistic_dif_models(results_dict, c_list, penalty_list, x_train, y_train, x_test):
    results_list = []
    for c in c_list:
        for penalty in penalty_list:
            params = 'logistic: ' + str(c) + ', ' + str(penalty)
            y_pred = logistic_pred(x_test, x_train, y_train, c, penalty)
            results_dict[params] = y_pred


def svm_pred(data_set, x_train, y_train, c, ker):
    svm = SVC(C=c, kernel=ker)
    svm.fit(x_train, y_train)
    return svm.predict(data_set)


def svm_dif_models(results_dict, c_list, ker_list, x_train, y_train, x_test):
    results_list = []
    for c in c_list:
        for kernel in ker_list:
            params = 'SVM: ' + str(c) + ', ' + str(kernel)
            y_pred = svm_pred(x_test, x_train, y_train, c, kernel)
            results_dict[params] = y_pred


def rf_pred(data_set, x_train, y_train, n_estimators, criterion, max_features):
	rf = RandomForestClassifier(n_estimators = n_estimators, criterion = criterion, max_features = max_features)
	rf.fit(x_train, y_train)
	return rf.predict(data_set)


def rf_dif_models(results_dict, n_estimators_list, criterion_list, max_features_list, x_train, y_train, x_test):
	results_list = []
	for n_estimators in n_estimators_list:
		for criterion in criterion_list:
			for max_feature in max_features_list:
				params = 'RF: ' + str(n_estimators) + ', ' + str(criterion) + ', ' + str(max_feature)
				y_pred = rf_pred(x_test, x_train, y_train, n_estimators, criterion, max_feature)
				results_dict[params] = y_pred

def bag_pred(data_set, x_train, y_train, n_estimators, max_features):
	bag = BaggingClassifier(n_estimators = n_estimators, max_features = max_features)
	bag.fit(x_train, y_train)
	return bag.predict(data_set)

def bag_dif_models(results_dict, n_estimators_list, max_features_list, x_train, y_train, x_test):
	results_list = []
	for n_estimators in n_estimators_list:
		for max_feature in max_features_list:
			params = 'Bagging: ' + str(n_estimators) + ', ' + str(max_feature)
			y_pred = bag_pred(x_test, x_train, y_train, n_estimators, max_feature)
			results_dict[params] = y_pred


def boost_pred(data_set, x_train, y_train, n_estimators):
	boost = AdaBoostClassifier(n_estimators = n_estimators)
	boost.fit(x_train, y_train)
	return boost.predict(data_set)


def boost_dif_models(results_dict, n_estimators_list, x_train, y_train, x_test):
	results_list = []
	for n_estimators in n_estimators_list:
		params = 'Boosting: ' + str(n_estimators) 
		y_pred = boost_pred(x_test, x_train, y_train, n_estimators)
		results_dict[params] = y_pred


def present_results(y_test, predictions):
    results_list = []
    for k, v in predictions.items():
        inter_list = [k, accuracy(v, y_test), 
                                 precision(v, y_test),
                                 precision_top(v, y_test, 0.01),
                                 precision_top(v, y_test, 0.02),
                                 precision_top(v, y_test, 0.05),
                                 precision_top(v, y_test, 0.1),
                                 precision_top(v, y_test, 0.2),
                                 precision_top(v, y_test, 0.3),
                                 precision_top(v, y_test, 0.5), 
                                 recall(v, y_test),
                                 recall_top(v, y_test, 0.01),
                                 recall_top(v, y_test, 0.02),
                                 recall_top(v, y_test, 0.05),
                                 recall_top(v, y_test, 0.1),
                                 recall_top(v, y_test, 0.2),
                                 recall_top(v, y_test, 0.3),
                                 recall_top(v, y_test, 0.5),
                                 f1(v, y_test)]
        if k[:6] != 'd_tree': 
        	inter_list.append(roc_auc(v, y_test))
        else:
        	inter_list.append('ND')
        results_list. append(inter_list)
    df = pd.DataFrame(results_list)
    df.columns = ['Model', 'Accuracy', 'Precision', 'Precision top 1%', 'Precision top 2%', 
                'Precision top 5%', 'Precision top 10%', 'Precision top 20%', 'Precision top 30%', 
                'Precision top 50%', 'Recall','Recall top 1%', 'Recall top 2%', 'Recall top 5%', 'Recall top 10%',
                'Recall top 20%', 'Recall top 30%', 'Recall top 50%', 'F 1', 'ROC AUC']
    return df

def present_results_simp(y_test, predictions):
    results_list = []
    for k, v in predictions.items():
        inter_list = [k, accuracy(v, y_test), 
                                 precision(v, y_test),
                                 precision_top(v, y_test, 0.01),
                                 precision_top(v, y_test, 0.02),
                                 precision_top(v, y_test, 0.05),
                                 precision_top(v, y_test, 0.1),
                                 precision_top(v, y_test, 0.2),
                                 precision_top(v, y_test, 0.3),
                                 precision_top(v, y_test, 0.5), 
                                 recall(v, y_test),
                                 recall_top(v, y_test, 0.01),
                                 recall_top(v, y_test, 0.02),
                                 recall_top(v, y_test, 0.05),
                                 recall_top(v, y_test, 0.1),
                                 recall_top(v, y_test, 0.2),
                                 recall_top(v, y_test, 0.3),
                                 recall_top(v, y_test, 0.5),
                                 f1(v, y_test)]
        results_list. append(inter_list)
    df = pd.DataFrame(results_list)
    df.columns = ['Model', 'Accuracy', 'Precision', 'Precision top 1%', 'Precision top 2%', 
                'Precision top 5%', 'Precision top 10%', 'Precision top 20%', 'Precision top 30%', 
                'Precision top 50%', 'Recall','Recall top 1%', 'Recall top 2%', 'Recall top 5%', 'Recall top 10%',
                'Recall top 20%', 'Recall top 30%', 'Recall top 50%', 'F 1']
    return df

