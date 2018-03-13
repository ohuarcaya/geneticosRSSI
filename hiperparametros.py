
def adaboost_param():
    parameters = {
        'AdaBoostClassifier__algorithm' : ['SAMME', 'SAMME.R'],
        'AdaBoostClassifier__n_estimators': [50, 100, 500],
        'AdaBoostClassifier__learning_rate': [0.01, 0.1, 1.0, 2.0]
    }
    return parameters


def voting_param():
    parameters = {
        'VotingClassifier__voting': ['hard', 'soft']
    }
    return parameters


def gradientboosting_param():
    parameters = {
        'GradientBoostingClassifier__n_estimators': [100, 200, 300],
        'GradientBoostingClassifier__max_depth': [3, 6, 9],
        'GradientBoostingClassifier__learning_rate': [0.01, 0.1, 0.3],
    }
    return parameters


def extratrees_param():       
    parameters = {
        'ExtraTreesClassifier__n_estimators': [10, 12, 15, 18, 20],
        'ExtraTreesClassifier__criterion': ['gini', 'entropy'],
        'ExtraTreesClassifier__min_samples_leaf': [1, 2, 3, 4, 5],
        'ExtraTreesClassifier__max_leaf_nodes': [3, 5, 7, 9, None],
        'ExtraTreesClassifier__max_depth': [2, 3, 4, 5, None],
        'ExtraTreesClassifier__max_features' : [None, 'sqrt', 'log2'],
        'ExtraTreesClassifier__class_weight' : ['balanced_subsample', None, 'balanced']
    }
    return parameters


def randomforest_param():
    parameters = {
        'RandomForestClassifier__n_estimators': [5, 10, 15, 30],
        'RandomForestClassifier__criterion': ['gini', 'entropy'],
        'RandomForestClassifier__warm_start': [True,False],
        'RandomForestClassifier__max_features' : [None, 'sqrt', 'log2'],
        'RandomForestClassifier__class_weight' : ['balanced_subsample', None, 'balanced'],
        'RandomForestClassifier__max_depth': [2, 5, 10, 20]
    }    
    return parameters


def decisiontree_param():
    parameters = {
        'DecisionTreeClassifier__criterion': ['gini','entropy'],
        'DecisionTreeClassifier__splitter': ['best','random'],
        'DecisionTreeClassifier__max_features': ['sqrt','log2', None],
        'DecisionTreeClassifier__class_weight' : [None, 'balanced'],
        'DecisionTreeClassifier__max_depth': [2, 3, 10, 50, 100]
    }
    return parameters


def lda_param():
    parameters = {
        'LinearDiscriminantAnalysis__solver': ['svd', 'lsqr', 'eigen']
    }
    return parameters


def svc_param():
    parameters = {
        'SVC__kernel': ['linear','poly', 'rbf','sigmoid'],
        'SVC__shrinking': [True, False],
        'SVC__C': [1, 2, 5, 10],
        'SVC__degree': [1, 2, 3, 4, 5],
        'SVC__decision_function_shape': ['ovr']
    }
    return parameters


def knn_param():
    parameters = {
        'KNeighborsClassifier__n_neighbors': [1, 3, 5, 7, 9, 11],
        'KNeighborsClassifier__weights': ['uniform','distance'],
        'KNeighborsClassifier__algorithm': ['ball_tree','kd_tree','brute']
    }
    return parameters


def logistic_param():
    parameters = {
        'LogisticRegression__solver': ['newton-cg', 'lbfgs', 'sag'],
        'LogisticRegression__warm_start': [True, False],        
        'LogisticRegression__multi_class' : ['ovr', 'multinomial'],
        'LogisticRegression__C' : [0.8, 0.9, 1.0, 1.1, 1.2]
    }
    return parameters


def naivebayes_param():
    parameters = {
        'GaussianNB__priors': [None]
    }
    return parameters


def mlperceptron_param():
    parameters = {
        'MLPClassifier__hidden_layer_sizes': [10, 20, 50, 100],
        'MLPClassifier__activation': ['identity', 'logistic', 'tanh', 'relu'],
        'MLPClassifier__solver' : ['lbfgs', 'sgd', 'adam'],
        'MLPClassifier__learning_rate' : ['constant', 'invscaling', 'adaptive'],
        'MLPClassifier__alpha' : np.logspace(-5, 3, 5)
    }
    return parameters