'''
Auxiliary functions for cleaning data
Assignment 3 
CAPP 30254  Machine Learning for Public Policy

Author: Jesus I. Ramirez Franco
''' 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def select_features(df, list_of_features):
	return df[list_of_features]

def delete_features(df, list_of_features):
	return df.drop(list_of_features, axis=1)

def between_timestamp(df, var, upper_boundary, lower_boundary):
	return df[(df[var] >= pd.to_datetime(lower_boundary)) & (df[var] < pd.to_datetime(upper_boundary))]

def split(df, y, test_size):
    Y = df[y]
    X = df.drop(y, axis=1)
    return train_test_split(X, Y, test_size=test_size)


def drop_na(df, how, axis = 0):
    df.dropna(axis = axis, how=how, inplace = True)


def fill_na(df, list_of_features, value):
    for feature in list_of_features:
        df[feature].fillna(value, inplace = True)


def limit_outliers(df, variable, upper_limit, lower_limit = None):
	if upper_limit:
		df[variable] = df[variable].apply(lambda x: upper_limit if x >= upper_limit else x)
	if lower_limit:
		df[variable] = df[variable].apply(lambda x: lower_limit if x <= lower_limit else x)


def discretize(df, variable):
	'''
	Discretize a continuous variable.
	Inputs:
		- df (Pandas data frame): data frame that contains the variable that we
		  want to convert.
		- variable (string): name of the variable we want to convert.
	Returns a modified data frame.
	'''
	df[variable] = df[variable].apply(lambda x: int(round(x,0)))

def binarize(df, variable, t):
	df[variable] = df[variable].apply(lambda x: 1 if x==t else 0)

def dummirize(df, list_of_features):
	df = pd.get_dummies(df, prefix = list_of_features, columns = list_of_features)
	return df

def labels_to_numeric(df, variable):
	le = LabelEncoder()
	le.fit(df[variable].unique())
	df[variable] = le.transform(df[variable])



#

#cohort_final['withdraw_reason'] = LabelEncoder().fit_transform(cohort_final['withdraw_reason'].astype('str'))
#cohort_final.head()
