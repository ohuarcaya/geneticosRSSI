import os
import time
import warnings
import numpy as np
import random as rnd
import pandas as pd
from collections import defaultdict

# Librería Genética
from deap import base, creator, tools, algorithms

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


def set_config():
    top_conf = {}
    top_conf['LogisticRegression']	=	np.array([4, 1, 2, 3, 1])-1    
    top_conf['LinearDiscriminantAnalysis']	=	np.array([6, 1, 3, 3, 5])-1
    top_conf['GaussianNB']	=	np.array([4, 1, 2, 6, 1])-1
    top_conf['MLPClassifier']	=	np.array([4, 1, 4, 6, 1])-1
    top_conf['SVC']	=	np.array([6, 1, 6, 6, 1])-1#np.array([6, 6, 3, 3, 5])-1 #3
    top_conf['DecisionTreeClassifier']	=	np.array([6, 1, 3, 6, 5])-1#np.array([6, 1, 3, 3, 5])-1 #2
    top_conf['KNeighborsClassifier']	=	np.array([6, 1, 3, 6, 1])-1
    top_conf['RandomForestClassifier']	=	np.array([6, 1, 3, 3, 5])-1
    top_conf['ExtraTreesClassifier']	=	np.array([6, 1, 3, 3, 5])-1
    top_conf['GradientBoostingClassifier']	=	np.array([6, 1, 3, 3, 5])-1
    top_conf['AdaBoostClassifier']	=	np.array([6, 1, 3, 6, 5])-1#np.array([6, 1, 3, 3, 1])-1#6
    top_conf['VotingClassifier']	=	np.array([6, 1, 3, 3, 5])-1
    return top_conf
    
def set_models():
    rs = 1
    models = []
    models.append(('LogisticRegression', LogisticRegression(C=0.9, multi_class='multinomial', solver='newton-cg', warm_start=True)))
    # LDA : Warning(Variables are collinear)
    models.append(('LinearDiscriminantAnalysis', LinearDiscriminantAnalysis(solver='eigen')))    
    models.append(('GaussianNB', GaussianNB(priors=None)))
    models.append(('MLPClassifier', MLPClassifier(hidden_layer_sizes=100, alpha=1e-05, activation='tanh', solver='adam', learning_rate='invscaling')))
    models.append(('SVC', SVC(random_state=rs, C=10, kernel='poly', decision_function_shape='ovo')))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier(random_state=rs, max_features=None, splitter='best', criterion='entropy', class_weight=None, max_depth=10)))
    models.append(('KNeighborsClassifier', KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='ball_tree')))
    models.append(('RandomForestClassifier', RandomForestClassifier(random_state=rs, max_depth=20, class_weight=None, max_features='sqrt', criterion='gini', warm_start=True, n_estimators=30)))
    models.append(('ExtraTreesClassifier', ExtraTreesClassifier(random_state=rs, min_samples_leaf=3, n_estimators=20, max_leaf_nodes=None, max_depth=None, class_weight=None, max_features='sqrt', criterion='gini')))
    models.append(('GradientBoostingClassifier',GradientBoostingClassifier(random_state=rs, max_features='sqrt', learning_rate=0.1, max_depth=3, n_estimators=100)))
    models.append(('AdaBoostClassifier', AdaBoostClassifier(DecisionTreeClassifier(random_state=rs, max_features=None, splitter='best', criterion='entropy', class_weight=None, max_depth=10), 
                                                            n_estimators=500, algorithm='SAMME', learning_rate=2.0, random_state=rs)))
    estimators = []
    estimators.append(("Voting_GradientBoostingClassifier",GradientBoostingClassifier(random_state=rs, max_features='sqrt', learning_rate=0.1, max_depth=3, n_estimators=100)))
    estimators.append(("Voting_ExtraTreesClassifier", ExtraTreesClassifier(random_state=rs, min_samples_leaf=3, n_estimators=20, max_leaf_nodes=None, max_depth=None, class_weight=None, max_features='sqrt', criterion='gini'))) 
    voting = VotingClassifier(estimators, voting='soft')
    models.append(('VotingClassifier', voting))
    return models

test_size = 0.2
num_folds = 10
seed = 7
frecuencias = []
names_ = ['Be01', 'Be02', 'Be03', 'Be04', 'Be05', 'Sector']
num_jobs=cpu_count()

frecuencias.append(pd.read_csv('sinFiltro/Tx_0x01'))#, names=names_))
frecuencias.append(pd.read_csv('sinFiltro/Tx_0x02'))#, names=names_))
frecuencias.append(pd.read_csv('sinFiltro/Tx_0x03'))#, names=names_))
frecuencias.append(pd.read_csv('sinFiltro/Tx_0x04'))#, names=names_))
frecuencias.append(pd.read_csv('sinFiltro/Tx_0x05'))#, names=names_))
frecuencias.append(pd.read_csv('sinFiltro/Tx_0x06'))#, names=names_))

# find distance error al 0.2%
def distance_error(estimator, X_train, y_train, X_test, y_test):
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 7)
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
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 7)
	score = 0
	scorer = "accuracy"
	name = str(estimator).split('(')[0]
	paramkey = name+str(np.int32(individual)+1)
	if paramkey in score_cache:
		score = score_cache[paramkey]
	else:
		print("Modeling ....",name)
		kfold = KFold(n_splits=10, shuffle=False)
		start_time = time.time()
		cv_results = cross_val_score(estimator, X_train, y_train, cv=kfold, scoring=scorer, n_jobs=-1)
		run_time = time.time() - start_time
		print('--- tiempo de cvs: %s segundos ---' %(run_time))
		#print(name,"  ",paramkey,"   ")
		#print(len(X),"  ",len(y),"   ", kfold)
		score = cv_results.mean()
		desv = cv_results.std()
		error = distance_error(estimator, X_train, y_train, X_test, y_test)
		score_cache[paramkey] = score
		dict_result = {'Modelo': name, 'Configuracion':np.int32(individual)+1, 'values': cv_results, 'Accuracy': score, 'stdAccuracy': desv, 'errorMetrico': np.mean(error), 'error': error, 'runtime': run_time }
		resultados.append(dict_result)
	return score


score_cache = {}
resultados = []
lista_resultados = []
indice_modelo = 8 #0-11

print('\nAsignando Modelo ... ')
estimadores = set_models()
#for name,model in estimadores:
name, model = estimadores[indice_modelo]

print('\nAsignando Configuracion ...')
configuraciones = set_config()
values = configuraciones[name]

print('\nGeting Results ...')
getAccuracy(frecuencias, values, model, score_cache, resultados)

topDf = pd.DataFrame(resultados)
l_name = ['Modelo', 'Configuracion', 'Accuracy', 'stdAccuracy', 'errorMetrico', 'runtime']
print(topDf[l_name])#.to_csv('resultadosTOP1_gs.csv', sep=',', index=False)
#display(topDf[l_name])