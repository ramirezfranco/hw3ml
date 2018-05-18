'''
Auxiliary functions to do classifiers
Assignment 3 
CAPP 30254  Machine Learning for Public Policy

Author: Jesus I. Ramirez Franco
''' 

def knn_pred(data_set, x_train, y_train, k, weight_type):
    knn = KNeighborsClassifier(n_neighbors=k, weights = weight_type)
    knn.fit(x_train, y_train)
    return knn.predict(data_set)

def knn_metrics_table(k_list, weights_list):
	results_list = []
	for k in k_list:
		for weight in weights_list:
			results_list.append([criterion, splitter, depth, accuracy(dec_tree.predict(x_train), y_train), 
					accuracy(dec_tree.predict(x_test), y_test)])
	df = pd.DataFrame(results_list)
	df.columns = ['criterion', 'splitter', 'max_depth', 'accuracy_train', 'accuracy_test']
	return df

def dtree_pred(data_set, x_train, y_train, criterion, splitter, depth):
	dec_tree = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=depth)
	dec_tree.fit(x_train, y_train)
	return dec_tree.predict(data_set)

def dtree_metrics_table(criterion_list, splitter_list, max_depth_list):
	results_list = []
	for criterion in criterion_list:
		for splitter in splitter_list:
			for depth in max_depth_list:
				dec_tree = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=depth)
				dec_tree.fit(x_train, y_train)
				results_list.append([criterion, splitter, depth, accuracy(dec_tree.predict(x_train), y_train), 
					accuracy(dec_tree.predict(x_test), y_test)])
	df = pd.DataFrame(results_list)
	df.columns = ['criterion', 'splitter', 'max_depth', 'accuracy_train', 'accuracy_test']
	return df


def logistic_pred(data_set, x_train, y_train, c, penalty):
	logireg = LogisticRegression(C=c, penalty=penalty)
    logireg.fit(x_train, y_train)
    return logireg.predict(data_set)

def svm_pred(data_set, x_train, y_train, c, ker, deg, gam, coef, prob):
	svm = svc(C=c, kernel=ker, degree=deg, gamma=gam, coef0=coef, probability=prob)
	svm.fit(x_train, y_train)
	return svm.predict(data_set)


