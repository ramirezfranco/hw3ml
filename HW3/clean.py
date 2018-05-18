'''
Auxiliary functions for cleaning data
Assignment 3 
CAPP 30254  Machine Learning for Public Policy

Author: Jesus I. Ramirez Franco
''' 
import pandas as pd

def select_features(df, list_of_features):
	return df[list_of_features]

def between_timestamp(df, var, upper_boundary, lower_boundary):
	return df[(df[var] >= pd.to_datetime(lower_boundary)) & (df[var] <= pd.to_datetime(upper_boundary))]
