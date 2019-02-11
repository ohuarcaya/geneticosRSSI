"""
    Librerias
"""
import os
import time
import warnings
import numpy as np
import random as rnd
import pandas as pd
from collections import defaultdict

# Librería Genética
#from deap import base, creator, tools, algorithms

from sklearn.utils import shuffle
# Subfunciones de estimadores
from sklearn.base import clone
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py][30]
from sklearn.base import is_classifier
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py][535]
from sklearn.model_selection._validation import _fit_and_score
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_validation.py][346]
from sklearn.model_selection._search import BaseSearchCV
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_search.py][386]
from sklearn.model_selection._search import check_cv
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_split.py][1866]
from sklearn.model_selection._search import _check_param_grid
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_search.py][343]
from sklearn.metrics.scorer import check_scoring
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/scorer.py][250]
from sklearn.utils.validation import _num_samples
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/validation.py][105]
from sklearn.utils.validation import indexable
# [https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/validation.py][208]
from multiprocessing import Pool, Manager, cpu_count

# Selección para estimadores
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Metricas para estimadores
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

# Estimadores
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import FunctionTransformer

#Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

#Ensembles algorithms
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt

"""
    Funciones
"""

def set_config():
    top_conf = {}
    top_conf['LogisticRegression']  =   np.array([4, 1, 2, 3, 1])-1    
    top_conf['LinearDiscriminantAnalysis']  =   np.array([6, 1, 3, 3, 5])-1
    top_conf['GaussianNB']  =   np.array([4, 1, 2, 6, 1])-1
    top_conf['MLPClassifier']   =   np.array([4, 1, 4, 6, 1])-1
    top_conf['SVC'] =   np.array([6, 1, 6, 6, 1])-1#np.array([6, 6, 3, 3, 5])-1 #3
    top_conf['DecisionTreeClassifier']  =   np.array([6, 1, 3, 6, 5])-1#np.array([6, 1, 3, 3, 5])-1 #2
    top_conf['KNeighborsClassifier']    =   np.array([6, 1, 3, 6, 1])-1
    top_conf['RandomForestClassifier']  =   np.array([6, 1, 3, 3, 5])-1
    top_conf['ExtraTreesClassifier']    =   np.array([6, 1, 3, 3, 5])-1
    top_conf['GradientBoostingClassifier']  =   np.array([6, 1, 3, 3, 5])-1
    top_conf['AdaBoostClassifier']  =   np.array([6, 1, 3, 6, 5])-1#np.array([6, 1, 3, 3, 1])-1 #6
    top_conf['VotingClassifier']    =   np.array([6, 1, 3, 3, 5])-1
    return top_conf
   

def set_models():
    rs = 1
    models = []
    models.append(('LogisticRegression', LogisticRegression()))
    # LDA : Warning(Variables are collinear)
    models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis()))    
    models.append(('GaussianNB', GaussianNB()))
    models.append(('MLPClassifier', MLPClassifier()))
    models.append(('SVC', SVC(random_state=rs)))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier(random_state=rs)))
    models.append(('KNeighborsClassifier', KNeighborsClassifier()))
    models.append(('RandomForestClassifier', RandomForestClassifier(random_state=rs)))
    # Bagging and Boosting
    # models.append(('ExtraTreesClassifier', ExtraTreesClassifier(n_estimators=150)))
    models.append(('ExtraTreesClassifier', ExtraTreesClassifier(random_state=rs)))
    # models.append(('AdaBoostClassifier', AdaBoostClassifier(DecisionTreeClassifier())))
    models.append(('GradientBoostingClassifier',GradientBoostingClassifier(random_state=rs)))
    models.append(('AdaBoostClassifier', AdaBoostClassifier(DecisionTreeClassifier(random_state=rs),random_state=rs)))
    # models.append(('GradientBoostingClassifier', GradientBoostingClassifier()))
    # Voting
    estimators = []
    estimators.append(("Voting_GradientBoostingClassifier", GradientBoostingClassifier(random_state=rs)))
    estimators.append(("Voting_ExtraTreesClassifier", ExtraTreesClassifier(random_state=rs)))
    voting = VotingClassifier(estimators)
    models.append(('VotingClassifier', voting))
    return models


test_size = 0.2
num_folds = 10
seed = 7
frecuencias = []
names_ = ['Be01', 'Be02', 'Be03', 'Be04', 'Be05', 'Sector']

frecuencias.append(pd.read_csv('sinFiltro/Tx_0x01'))#, names=names_))
frecuencias.append(pd.read_csv('sinFiltro/Tx_0x02'))#, names=names_))
frecuencias.append(pd.read_csv('sinFiltro/Tx_0x03'))#, names=names_))
frecuencias.append(pd.read_csv('sinFiltro/Tx_0x04'))#, names=names_))
frecuencias.append(pd.read_csv('sinFiltro/Tx_0x05'))#, names=names_))
frecuencias.append(pd.read_csv('sinFiltro/Tx_0x06'))#, names=names_))
"""
frecuencias.append(pd.read_csv('Filtrado/LocalizationNew_Tx1.csv', names=names_))
frecuencias.append(pd.read_csv('Filtrado/LocalizationNew_Tx2.csv', names=names_))
frecuencias.append(pd.read_csv('Filtrado/LocalizationNew_Tx3.csv', names=names_))
frecuencias.append(pd.read_csv('Filtrado/LocalizationNew_Tx4.csv', names=names_))
frecuencias.append(pd.read_csv('Filtrado/LocalizationNew_Tx5.csv', names=names_))
frecuencias.append(pd.read_csv('Filtrado/LocalizationNew_Tx6.csv', names=names_))
"""
num_jobs=cpu_count()
#estimadores = set_models()
#configuracion = set_models()
#salida = {}

# find distance error al 0.2%
def distance_error(estimator, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 7)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    # coord pred
    x1 = np.int32((y_pred + 2) % 3)
    y1 = np.int32((y_pred - 1) / 3)
    # coord real
    x2 = np.int32((y_test + 2) % 3)
    y2 = np.int32((y_test - 1) / 3)
    # pasar variacion a distancias metros
    vx = np.abs(x1 - x2)*1.5
    vy = np.abs(y1 - y2)*1.5
    #vx = vx*0.5 + (vx-1)*(vx>0)
    #vy = vy*0.5 + (vy-1)*(vy>0)
    # pitagoras
    err_distance = np.sqrt(vx*vx + vy*vy)
    return err_distance


#def _createDataset(frecuencias, values, seed = 7):
def _createDataset(frecuencias, values):
    from sklearn.utils import shuffle as shuff
    # crear dataset
    names_ = frecuencias[0].columns.values
    seed = 7
    # reestructuracion
    salida_final = pd.DataFrame(columns=names_)
    for sec in range(1,16):
        dataset = pd.DataFrame(columns=names_)
        corte = min([frecuencias[i][frecuencias[i]['Sector']==sec].shape[0] for i in values])
        #l = [frecuencias[i][frecuencias[i]['Sector']==sec].shape[0] for i in values]
        #corte = max(l)
        #tx=l.index(max(l))
        tx = 0
        dataset[names_[tx]] = dataset[names_[tx]].append(frecuencias[int(values[tx])][frecuencias[int(values[tx])]['Sector']==sec][:corte][names_[tx]])
        dataset = dataset.reset_index(drop=True)
        for tx in range(1,5):
            dataset[names_[tx]] = frecuencias[int(values[tx])][frecuencias[int(values[tx])]['Sector']==sec][:corte][names_[tx]].reset_index(drop=True)
        dataset[names_[tx+1]] = frecuencias[int(values[tx])][frecuencias[int(values[tx])]['Sector']==sec][:corte][names_[tx+1]].reset_index(drop=True)
        # join parts
        salida_final = salida_final.append(dataset)
    # shuffle dataset
    salida_final = shuff(salida_final, random_state=seed).reset_index(drop=True)
    salida_final = salida_final.apply(pd.to_numeric)
    # dataframe to X,y 
    X = salida_final[names_[:-1]]
    y = salida_final[names_[-1]]
    return X,y


# The problem to optimize
def getAccuracy( frecuencias, individual, estimator, score_cache, resultados ):
    X,y = _createDataset(frecuencias, individual)
    score = 0
    scorer = "accuracy"
    name = str(estimator).split('(')[0]
    paramkey = name+str(np.int32(individual)+1)
    if paramkey in score_cache:
        score = score_cache[paramkey]
    else:
        print("Modeling ....",name)
        kfold = KFold(n_splits=10, shuffle=False)
        cv_results = cross_val_score(estimator, X, y, cv=kfold, scoring=scorer)
        #print(name,"  ",paramkey,"   ")
        #print(len(X),"  ",len(y),"   ", kfold)
        score = cv_results.mean()
        desv = cv_results.std()
        error = distance_error(estimator, X, y)
        score_cache[paramkey] = score
        dict_result = {'Modelo': name, 'Configuracion':np.int32(individual)+1, 'values': cv_results, 'Accuracy': score, 'stdAccuracy': desv, 'errorMetrico': np.mean(error), 'error': error }
        resultados.append(dict_result)
    return score





"""
   Mas Funciones 
"""

"""
from __future__ import print_function
import tools

import warnings
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
warnings.filterwarnings("ignore", category=DeprecationWarning)

from collections import OrderedDict
from time import time
from plotly.offline.offline import _plot_html
from scipy.stats import randint
from scipy.stats import expon

# from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

This module provides ideas for improving some machine learning algorithms.
"""

def adaboost_param():
    parameters = {
        # 'selector__extraTC__n_estimators': [10],
        # 'selector__extraTC__n_estimators': [10, 15],
        # # 'selector__extraTC__criterion': ['entropy'],
        # 'selector__extraTC__criterion': ['gini', 'entropy'],
        # 'selector__extraTC__n_jobs': [-1],
        # 'selector__pca__svd_solver': ['randomized'],
        #'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
        # 'selector__pca__whiten': [True],
        #'selector__pca__whiten': [True,False],
        'AdaBoostClassifier__algorithm' : ['SAMME', 'SAMME.R'],
        'AdaBoostClassifier__n_estimators': [50, 100, 500],
        'AdaBoostClassifier__learning_rate': [0.01, 0.1, 1.0, 2.0]
    }
    return parameters


def voting_param():
    parameters = {
        # 'selector__extraTC__n_estimators': [10],
        # 'selector__extraTC__n_estimators': [10, 15],
        # # 'selector__extraTC__criterion': ['entropy'],
        # 'selector__extraTC__criterion': ['gini', 'entropy'],
        # 'selector__extraTC__n_jobs': [-1],
        # 'selector__pca__svd_solver': ['randomized'],
        #'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
        # 'selector__pca__whiten': [True],
        #'selector__pca__whiten': [True,False],
        'VotingClassifier__voting': ['hard', 'soft']
    }
    return parameters


def gradientboosting_param():
    parameters = {
        # 'selector__extraTC__n_estimators': [10],
        # 'selector__extraTC__n_estimators': [10, 15],
        # # 'selector__extraTC__criterion': ['entropy'],
        # 'selector__extraTC__criterion': ['gini', 'entropy'],
        # 'selector__extraTC__n_jobs': [-1],
        # 'selector__pca__svd_solver': ['randomized'],
        #'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
        # 'selector__pca__whiten': [True],
        #'selector__pca__whiten': [True,False],
        'GradientBoostingClassifier__n_estimators': [100, 200, 300],
        'GradientBoostingClassifier__max_depth': [3, 6, 9],
        'GradientBoostingClassifier__learning_rate': [0.01, 0.1, 0.3],
        'GradientBoostingClassifier__max_features' : ["sqrt", "log2", None]#,
        #'GradientBoostingClassifier__loss' : ['deviance', 'exponential']
    }
    return parameters


def extratrees_param():       
    parameters = {
        # 'selector__extraTC__n_estimators': [10],
        # 'selector__extraTC__n_estimators': [10, 15],
        # 'selector__extraTC__criterion': ['gini', 'entropy'],
        # # 'selector__extraTC__criterion': ['entropy'],
        # 'selector__extraTC__n_jobs': [-1],
        # 'selector__pca__svd_solver': ['randomized'],
        #'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
        # 'selector__pca__whiten': [True],
        #'selector__pca__whiten': [True,False],
        'ExtraTreesClassifier__n_estimators': [10, 12, 15, 18, 20],
        'ExtraTreesClassifier__criterion': ['gini', 'entropy'],
        'ExtraTreesClassifier__min_samples_leaf': [1, 2, 3, 4, 5],
        #'ExtraTreesClassifier__min_samples_leaf': range(200,1001,200),
        'ExtraTreesClassifier__max_leaf_nodes': [3, 5, 7, 9, None],
        'ExtraTreesClassifier__max_depth': [2, 3, 4, 5, None],
        'ExtraTreesClassifier__max_features' : [None, 'sqrt', 'log2'],
        'ExtraTreesClassifier__class_weight' : ['balanced_subsample', None, 'balanced']
    }
    return parameters


def randomforest_param():
    parameters = {
        # 'selector__extraTC__n_estimators': [10],
        # 'selector__extraTC__n_estimators': [10, 15],
        # # 'selector__extraTC__criterion': ['entropy'],
        # 'selector__extraTC__criterion': ['gini', 'entropy'],
        # 'selector__extraTC__n_jobs': [-1],
        # 'selector__pca__svd_solver': ['randomized'],
        #'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
        # 'selector__pca__whiten': [True],
        #'selector__pca__whiten': [True,False],
        'RandomForestClassifier__n_estimators': [5, 10, 15, 30],
        'RandomForestClassifier__criterion': ['gini', 'entropy'],
        'RandomForestClassifier__warm_start': [True,False],
        'RandomForestClassifier__max_features' : [None, 'sqrt', 'log2'],
        # 'RandomForestClassifier__min_samples_leaf': [1,2,3,4,5],
        # 'RandomForestClassifier__max_leaf_nodes': [2,3,4,5],
        'RandomForestClassifier__class_weight' : ['balanced_subsample', None, 'balanced'],
        'RandomForestClassifier__max_depth': [2, 5, 10, 20]
    }    
    return parameters


def decisiontree_param():
    parameters = {
        # 'selector__extraTC__n_estimators':  [10],
        # 'selector__extraTC__n_estimators':  [10, 15],
        # # 'selector__extraTC__criterion': ['entropy'],
        # 'selector__extraTC__criterion': ['gini','entropy'],
        # 'selector__extraTC__n_jobs': [-1],
        # 'selector__pca__svd_solver': ['randomized'],
        #'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
        # 'selector__pca__whiten': [True],
        #'selector__pca__whiten': [True,False],
        'DecisionTreeClassifier__criterion': ['gini','entropy'],
        'DecisionTreeClassifier__splitter': ['best','random'],
        'DecisionTreeClassifier__max_features': ['sqrt','log2', None],
        # 'DecisionTreeClassifier__max_leaf_nodes': [2,3, None],
        'DecisionTreeClassifier__class_weight' : [None, 'balanced'],
        'DecisionTreeClassifier__max_depth': [2, 3, 10, 50, 100]
        # 'DecisionTreeClassifier__min_samples_leaf': [1,3,5, None]
    }
    return parameters


def lda_param():
    parameters = {
        # 'selector__extraTC__n_estimators':  [10],
        # 'selector__extraTC__n_estimators':  [10, 15],
        # # 'selector__extraTC__criterion': ['entropy'],
        # 'selector__extraTC__criterion': ['gini','entropy'],
        # 'selector__extraTC__n_jobs': [-1],
        # 'selector__pca__svd_solver': ['randomized'],
        #'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
        # 'selector__pca__whiten': [True],
        #'selector__pca__whiten': [True,False],
        'LinearDiscriminantAnalysis__solver': ['svd', 'lsqr', 'eigen']
    }
    return parameters


def svc_param():
    parameters = {
        # 'selector__extraTC__n_estimators':  [10],
        # 'selector__extraTC__n_estimators':  [10, 15],
        # # 'selector__extraTC__criterion': ['entropy'],
        # 'selector__extraTC__criterion': ['gini','entropy'],
        # 'selector__extraTC__n_jobs': [-1],
        #'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
        # 'selector__pca__svd_solver': ['randomized'],
        # 'selector__pca__whiten': [True],
        #'selector__pca__whiten': [True,False],
        'SVC__kernel': ['linear','poly', 'rbf','sigmoid'],
        # 'SVC__kernel': ['rbf'],
        'SVC__shrinking': [True, False],
        'SVC__C': [1, 2, 5, 10],
        'SVC__degree': [1, 2, 3, 4, 5],
        #'SVC__decision_function_shape': ['ovo','ovr']
        'SVC__decision_function_shape': ['ovr']
    }
    return parameters


def knn_param():
    parameters = {
        # 'selector__extraTC__n_estimators':  [10, 15],
        # # 'selector__extraTC__n_estimators':  [10],
        # 'selector__extraTC__criterion': ['gini','entropy'],
        # 'selector__extraTC__n_jobs': [-1],
        # 'selector__pca__svd_solver': ['randomized'],
        #'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
        # 'selector__pca__whiten': [True],
        #'selector__pca__whiten': [True,False],
        'KNeighborsClassifier__n_neighbors': [1, 3, 5, 7, 9, 11],
        'KNeighborsClassifier__weights': ['uniform','distance'],
        'KNeighborsClassifier__algorithm': ['ball_tree','kd_tree','brute']
        # 'KNeighborsClassifier__algorithm': ['auto']
    }
    return parameters


def logistic_param():
    parameters = {
        # 'selector__extraTC__n_estimators':  [10],
        # 'selector__extraTC__n_estimators':  [10, 15],
        # 'selector__extraTC__criterion': ['gini','entropy'],
        # 'selector__extraTC__n_jobs': [-1],
        # 'selector__pca__svd_solver': ['randomized'],
        #'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
        # 'selector__pca__whiten': [True],
        #'selector__pca__whiten': [True,False],
        #'LogisticRegression__penalty': ['l2'], #, 'l1'
        # 'LogisticRegression__solver': ['newton-cg','lbfgs','liblinear','sag'],
        'LogisticRegression__solver': ['newton-cg', 'lbfgs', 'sag'],
        'LogisticRegression__warm_start': [True, False],        
        'LogisticRegression__multi_class' : ['ovr', 'multinomial'],
        'LogisticRegression__C' : [0.8, 0.9, 1.0, 1.1, 1.2]
    }
    return parameters


def naivebayes_param():
    parameters = {
        # 'selector__extraTC__n_estimators':  [10],
        # 'selector__extraTC__n_estimators':  [10, 15],
        # 'selector__extraTC__criterion': ['gini','entropy'],
        # 'selector__extraTC__n_jobs': [-1],
        #'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
        # 'selector__pca__whiten': [True],
        #'selector__pca__whiten': [True,False]
        'GaussianNB__priors': [None]
    }
    return parameters


def mlperceptron_param():
    parameters = {
        # 'selector__extraTC__n_estimators':  [10],
        # 'selector__extraTC__n_estimators':  [10, 15],
        # 'selector__extraTC__criterion': ['gini','entropy'],
        # 'selector__extraTC__n_jobs': [-1],
        #'selector__pca__svd_solver': ['full', 'arpack', 'randomized'],
        #'selector__pca__whiten': [True,False],
        'MLPClassifier__hidden_layer_sizes': [10, 20, 50, 100],
        'MLPClassifier__activation': ['identity', 'logistic', 'tanh', 'relu'],
        'MLPClassifier__solver' : ['lbfgs', 'sgd', 'adam'],
        'MLPClassifier__learning_rate' : ['constant', 'invscaling', 'adaptive'],
        'MLPClassifier__alpha' : np.logspace(-5, 3, 5)
    }
    return parameters


def get_params(model):
    if model == 'AdaBoostClassifier':
        return adaboost_param()
    elif model == 'VotingClassifier':
        return voting_param()
    elif model == 'GradientBoostingClassifier':
        return gradientboosting_param()
    elif model == 'ExtraTreesClassifier':
        return extratrees_param()
    elif model == 'RandomForestClassifier':
        return randomforest_param()
    elif model == 'DecisionTreeClassifier':
        return decisiontree_param()
    elif model == 'LinearDiscriminantAnalysis':
        return lda_param()
    elif model == 'SVC':
        return svc_param()
    elif model == 'KNeighborsClassifier':
        return knn_param()
    elif model == 'LogisticRegression':
        return logistic_param()
    elif model == 'GaussianNB':
        return naivebayes_param()
    elif model == 'MLPClassifier':
        return mlperceptron_param()
    return None




### frecuencias
configuracion = set_config() #dict ['name']=configuracion
estimators = set_models() #list (...(name, model))

num_splits = 10
kfold = KFold(n_splits=num_splits, shuffle=False)

lista_reporte = []
lista_df = [0]*12
indice = 0
for name, model in estimators:
    # name, model     # nombre y modelo
    grilla_parametros = get_params(name)   # parametros
    grilla_parametros = dict((key.split(name+'__')[-1], value) for (key,value) in grilla_parametros.items())
    X, y = _createDataset(frecuencias, configuracion[name])
    grid_search_t = GridSearchCV(model, grilla_parametros, n_jobs = -1, verbose=1, cv = kfold)
    print('GridSearch for ',name)
    try:
        grid_search_t.fit(X, y)
        # por diccionario
        reporte = {}
        reporte['modelo'] = name
        reporte['best_score'] = round(grid_search_t.best_score_, 3)
        reporte['fits'] = len(grid_search_t.grid_scores_)*num_splits
        reporte['parametros'] = grid_search_t.best_params_
        lista_reporte.append(reporte)
        # Por dataframe
        dfcv = pd.DataFrame(grid_search_t.cv_results_)
        dfcv['values'] = pd.DataFrame(grid_search_t.grid_scores_)['cv_validation_scores']
        dfcv['Model'] = name
        column_head = ['Model', 'mean_test_score', 'std_test_score', 'mean_fit_time', 'mean_score_time', 'values', 'params']
        dfcv = dfcv[column_head].sort_values('mean_test_score', ascending=False)#.head(5)
        lista_df[indice] =dfcv
    except:
        print ('\n\n\nERROR CON ', name,'\n\n\n')
        #lista_reporte.append({})
        #lista_df[indice] = 1
    indice = indice + 1


fulldf = pd.DataFrame(lista_reporte).sort_values(['best_score'],ascending=False)
#topDf = pd.DataFrame(resultados)

fulldf[['modelo', 'best_score', 'fits', 'parametros']].to_csv('fullGridSearch.csv', sep=',', index=False)
#display(fulldf)

for i in range(len(lista_reporte)):
    df = lista_df[i].sort_values(['mean_test_score'],ascending=False)
    name = list(df['Model'])[0]
    df[['Model', 'mean_test_score', 'std_test_score', 'mean_fit_time', 'mean_score_time', 'params']].to_csv('_'+name+'_gs.csv', sep=',', index=False)


"""
**GradientBoostingClassifier        max_features='sqrt', learning_rate=0.1, max_depth=3, n_estimators=100
**VotingClassifier      voting='soft'
**AdaBoostClassifier        n_estimators=500, algorithm='SAMME', learning_rate=2.0
**RandomForestClassifier        max_depth=20, class_weight=None, max_features='sqrt', criterion='gini', warm_start=True, n_estimators=30
**AdaBoostClassifier        algorithm='SAMME.R', learning_rate=2.0, n_estimators=500
**KNeighborsClassifier      n_neighbors=5, weights='distance', algorithm='ball_tree'
**DecisionTreeClassifier        max_features=None, splitter='best', criterion='entropy', class_weight=None, max_depth=10
**GaussianNB        priors=None
**DecisionTreeClassifier        class_weight=None, splitter='best', max_features=None, max_depth=10, criterion='entropy'
**SVC       C=10, kernel='poly', decision_function_shape='ovo'
**SVC       decision_function_shape='ovo', kernel='rbf', C=1
**ExtraTreesClassifier      max_depth=5, class_weight='balanced_subsample', min_samples_leaf=1, max_features='sqrt', max_leaf_nodes=9, criterion='gini', n_estimators=15
**LinearDiscriminantAnalysis        solver='eigen'
**LogisticRegression        C=0.9, multi_class='multinomial', solver='newton-cg', warm_start=True
**MLPClassifier     hidden_layer_sizes=100, alpha=1.0000000000000001e-05, activation='tanh', solver='adam', learning_rate='invscaling'



"""