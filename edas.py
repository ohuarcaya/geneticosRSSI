import os
import time
import warnings
import numpy as np
import random as rnd
import pandas as pd
from collections import defaultdict

# Librería Genética
from deap import base, creator, tools, algorithms

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
from multiprocessing import Pool

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

def _get_param_types_maxint(params):
	params_data = list(params.items())  # name_values
	params_type = [isinstance(params[key][0], float) + 1 for key in params.keys()]  # gene_type
	params_size = [len(params[key]) - 1 for key in params.keys()]  # maxints
	return params_data, params_type, params_size


def _initIndividual(pcls, maxints):
	"""[Iniciar Individuo]
	Arguments:
		pcls {[creator.Individual]} -- [Iniciar individuo con indices aleatorios]
		maxints {[params_size]} -- [lista de máximos índices]
	Returns:
		[creator.Individual] -- [Creación de individuo]
	"""
	part = pcls(rnd.randint(0, maxint) for maxint in maxints)
	return part


def _mutIndividual(individual, maxints, prob_mutacion):
	"""[Mutación Individuo]
	Arguments:
		individual {[creator.Individual]} -- [Individuo de población]
		maxints {[lista]} -- [lista de máximos índices]
		prob_mutacion {[float]} -- [probabilidad de mutación del gen]
	Returns:
		[creator.Individual] -- [Individuo mutado]
	"""
	for i in range(len(maxints)):
		if rnd.random() < prob_mutacion:
			individual[i] = rnd.randint(0, maxints[i])
	return individual,


def _cxIndividual(ind1, ind2, prob_cruce, gene_type):
	"""[Cruce de Individuos]
	Arguments:
		ind1 {[creator.Individual]} -- [Individuo 1]
		ind2 {[creator.Individual]} -- [Individuo 2]
		indpb {[float]} -- [probabilidad de emparejar]
		gene_type {[list]} -- [tipos de dato de los parámetros, CATEGORICO o NUMERICO]
	Returns:
		[creator.Individual,creator.Individual] -- [nuevos Individuos]
	"""
	CATEGORICO = 1  # int o str
	NUMERICO = 2  # float
	for i in range(len(ind1)):
		if rnd.random() < prob_cruce:
			if gene_type[i] == CATEGORICO:
				ind1[i], ind2[i] = ind2[i], ind1[i]
			else:
				sorted_ind = sorted([ind1[i], ind2[i]])
				ind1[i] = rnd.randint(sorted_ind[0], sorted_ind[1])
				ind2[i] = rnd.randint(sorted_ind[0], sorted_ind[1])
	return ind1, ind2


def _individual_to_params(individual, name_values):
	"""[Set de parámetro según individuo]
	Arguments:
		individual {[creator.Individual]} -- [individuo]
		name_values {[list]} -- [lista de parámetros, params_data]
	Returns:
		[diccionario] -- [parámetros del individuo]
	"""
	return dict((name, values[gene]) for gene, (name, values) in zip(individual, name_values))


def _evalFunction(individual, name_values, X, y, scorer, cv, uniform, fit_params,
				verbose=0, error_score='raise', score_cache={}):
	"""[Evaluación del modelo]
	Arguments:
		individual {[creator.Individual]} -- [Individuo]
		name_values {[list]} -- [parámetros en general]
		X {[array]} -- [Input]
		y {[array]} -- [Output]
		scorer {[string]} -- [Parámetro de evaluación, precisión]
		cv {[int | cross-validation]} -- [Especificación de los folds]
		uniform {[boolean]} -- [True hace que la data se distribuya uniformemente en los folds]
		fit_params {[dict | None]} -- [parámetros para estimator.fit]
	Keyword Arguments:
		verbose {integer} -- [Mensajes de descripción] (default: {0})
		error_score {numerico} -- [valor asignado si ocurre un error en fitting] (default: {'raise'})
		score_cache {dict} -- [description] (default: {{}})
	"""
	parameters = _individual_to_params(individual, name_values)
	score = 0
	n_test = 0
	paramkey = str(individual)
	if paramkey in score_cache:
		score = score_cache[paramkey]
	else:
		for train, test in cv.split(X, y):
			_score = _fit_and_score(estimator=individual.est, X=X, y=y, scorer=scorer,
						train=train, test=test, verbose=verbose,
						parameters=parameters, fit_params=fit_params,
						error_score=error_score)[0]
			if uniform:
				score += _score * len(test)
				n_test += len(test)
			else:
				score += _score
				n_test += 1
		assert n_test > 0, "No se completo el fitting, Verificar data."
		score /= float(n_test)
		score_cache[paramkey] = score
	return (score,)

class GeneticSearchCV:
	def __init__(self, estimator, params, scoring=None, cv=4,
				refit=True, verbose=False, population_size=50,
				gene_mutation_prob=0.1, gene_crossover_prob=0.5,
				tournament_size=3, generations_number=10, gene_type=None,
				n_jobs=1, uniform=True, error_score='raise',
				fit_params={}):
		# Parámetros iniciales
		self.estimator = estimator
		self.params = params
		self.scoring = scoring
		self.cv = cv
		self.refit = refit
		self.verbose = verbose
		self.population_size = population_size
		self.gene_mutation_prob = gene_mutation_prob
		self.gene_crossover_prob = gene_crossover_prob
		self.tournament_size = tournament_size
		self.generations_number = generations_number
		self.gene_type = gene_type
		self.n_jobs = n_jobs
		self.uniform = uniform
		self.error_score = error_score
		self.fit_params = fit_params
		# Parámetros adicionales
		self._individual_evals = {}
		self.all_history_ = None
		self.all_logbooks_ = None
		self._cv_results = None
		self.best_score_ = None
		self.best_params_ = None
		self.scorer_ = None
		self.score_cache = {}
		# Fitness [base.Fitness], objetivo 1
		creator.create("FitnessMax", base.Fitness, weights=(1.0,))
		# Individuo [list], parámetros:est, FinessMax
		creator.create("Individual", list, est=clone(self.estimator), fitness=creator.FitnessMax)
	@property
	def cv_results_(self):
		if self._cv_results is None:
			out = defaultdict(list)
			gen = self.all_history_
			# Get individuals and indexes, their list of scores,
			# and additionally the name_values for this set of parameters
			idxs, individuals, each_scores = zip(*[(idx, indiv, np.mean(indiv.fitness.values))
											for idx, indiv in list(gen.genealogy_history.items())
											if indiv.fitness.valid and not np.all(np.isnan(indiv.fitness.values))])
			name_values, _, _ = _get_param_types_maxint(params)
			# Add to output
			#out['param_index'] += [p] * len(idxs)
			out['index'] += idxs
			out['params'] += [_individual_to_params(indiv, name_values) for indiv in individuals]
			out['mean_test_score'] += [np.nanmean(scores) for scores in each_scores]
			out['std_test_score'] += [np.nanstd(scores) for scores in each_scores]
			out['min_test_score'] += [np.nanmin(scores) for scores in each_scores]
			out['max_test_score'] += [np.nanmax(scores) for scores in each_scores]
			out['nan_test_score?'] += [np.any(np.isnan(scores)) for scores in each_scores]
			self._cv_results = out
		return self._cv_results
	@property
	def best_index_(self):
		return np.argmax(self.cv_results_['max_test_score'])
	# fit y refit general
	def fit(self, X, y):
		self.best_estimator_ = None
		self.best_mem_score_ = float("-inf")
		self.best_mem_params_ = None
		_check_param_grid(self.params)
		self._fit(X, y, self.params)
		if self.refit:
			self.best_estimator_ = clone(self.estimator)
			self.best_estimator_.set_params(**self.best_mem_params_)
			if self.fit_params is not None:
				self.best_estimator_.fit(X, y, **self.fit_params)
			else:
				self.best_estimator_.fit(X, y)
	# fit individual
	def _fit(self, X, y, parameter_dict):
		self._cv_results = None  # Indicador de necesidad de actualización
		self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)
		n_samples = _num_samples(X)
		# verificar longitudes x,y 
		if _num_samples(y) != n_samples:
			raise ValueError('Target [y], data [X] no coinciden')
		self.cv = check_cv(self.cv, y=y, classifier=is_classifier(self.estimator))
		toolbox = base.Toolbox()
		# name_values = lista de parametros, gene_type = [1:categorico; 2:numérico], maxints = size(parametros)
		name_values, self.gene_type, maxints = _get_param_types_maxint(parameter_dict)
		if self.verbose:
			print("Tipos: %s, rangos: %s" % (self.gene_type, maxints))
		# registro de función Individuo
		toolbox.register("individual", _initIndividual, creator.Individual, maxints=maxints)
		# registro de función Población
		toolbox.register("population", tools.initRepeat, list, toolbox.individual)
		# Paralelísmo, create pool
		if not isinstance(self.n_jobs, int):
			self.n_jobs=1
		pool = Pool(self.n_jobs)
		toolbox.register("map", pool.map)
		# registro de función Evaluación
		toolbox.register("evaluate", _evalFunction,
						name_values=name_values, X=X, y=y,
						scorer=self.scorer_, cv=self.cv, uniform=self.uniform, verbose=self.verbose,
						error_score=self.error_score, fit_params=self.fit_params,
						score_cache=self.score_cache)
		# registro de función Cruce
		toolbox.register("mate", _cxIndividual, prob_cruce=self.gene_crossover_prob, gene_type=self.gene_type)
		# registro de función Mutación
		toolbox.register("mutate", _mutIndividual, prob_mutacion=self.gene_mutation_prob, maxints=maxints)
		# registro de función Selección
		toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
		# Creación de Población
		pop = toolbox.population(n=self.population_size)
		# Mejor Individuo que ha existido
		hof = tools.HallOfFame(1)
		# Stats
		stats = tools.Statistics(lambda ind: ind.fitness.values)
		stats.register("avg", np.nanmean)
		stats.register("min", np.nanmin)
		stats.register("max", np.nanmax)
		stats.register("std", np.nanstd)
		# Genealogía
		hist = tools.History()
		# Decoración de operadores de variaznza
		toolbox.decorate("mate", hist.decorator)
		toolbox.decorate("mutate", hist.decorator)
		hist.update(pop)
		# Posibles combinaciones
		if self.verbose:
			print('--- Evolve in {0} possible combinations ---'.format(np.prod(np.array(maxints) + 1)))
		pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
										ngen=self.generations_number, stats=stats,
										halloffame=hof, verbose=self.verbose)
		#pop, logbook = algorithms.eaGenerateUpdate(toolbox,
		#								ngen=self.generations_number, stats=stats,
		#								halloffame=hof, verbose=self.verbose)
		# Save History
		self.all_history_ = hist
		self.all_logbooks_ = logbook
		# Mejor score y parametros
		current_best_score_ = hof[0].fitness.values[0]
		current_best_params_ = _individual_to_params(hof[0], name_values)
		if self.verbose:
			print("Best individual is: %s\nwith fitness: %s" % (
				current_best_params_, current_best_score_))
		if current_best_score_ > self.best_mem_score_:
			self.best_mem_score_ = current_best_score_
			self.best_mem_params_ = current_best_params_
		# fin paralelización, close pool
		pool.close()
		pool.join()
		self.best_score_ = current_best_score_
		self.best_params_ = current_best_params_

class EstimatorSelection:
	"""[Clase de Estimación Lineal]
	init: 	inicia los modelos, parámetros de entrada y 
			grid_searches, resultados que se pueden obtener con get
	fit:	Entrena los estimadores linealmente y se actualiza grid_searches
			como diccionario de resultados por modelo
	score:	Toma los resultados de todas las estimaciones y las une
			para hacer un analisis comparativo
	Fuente:	http://www.codiply.com/blog/hyperparameter-grid-search-across-multiple-models-in-scikit-learn/
	"""

	def __init__(self, models, params):
		if not set(models.keys()).issubset(set(params.keys())):
			missing_params = list(set(models.keys()) - set(params.keys()))
			raise ValueError("Some estimators are missing parameters: %s" % missing_params)
		self.models = models
		self.params = params
		self.keys = models.keys()
		self.grid_searches = {}
		self.result = {}
		self.time_model ={} 

	def fit(self, X, y, cv=KFold(n_splits=5), n_jobs=1, verbose=1, scoring=None, refit=False,
			population_size=50, gene_mutation_prob=0.10, gene_crossover_prob=0.5,
			tournament_size=3, generations_number=10):
		for key in self.keys:
			print("Running GeneticSearch for %s." % key)
			model = self.models[key]
			params = self.params[key]
			start = time.time()
			gs = GeneticSearchCV(model, params, cv=cv, n_jobs=n_jobs,
							verbose=verbose, scoring=scoring, refit=refit)
			self.grid_searches[key] = gs.fit(X, y)
			end = time.time()
			self.time_model[key] = [end-start]

	def score(self):
		for estimator in self.keys:
			d = {}
			d['.Accuracy'] = self.grid_searches[estimator].cv_results_['mean_test_score']
			d['.Error'] = self.grid_searches[estimator].cv_results_['std_test_score']
			df1 = pd.DataFrame(d)
			d = self.grid_searches[estimator].cv_results_['params']
			df2 = pd.DataFrame(list(d))
			self.result[estimator] = pd.concat([df1, df2], axis=1).fillna(' ')
		return pd.concat(self.result).fillna(' ')



# Lectura de los datos y separación
dataset = pd.read_csv("raspberry/TransmisionInt.csv")
validation_size = 0.20
X = dataset.values[:, 0:dataset.shape[1] - 1].astype(float)
Y = dataset.values[:, dataset.shape[1] - 1]
# Split randomizando los datos
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=validation_size)

# Modelos para el Test
models = { 
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'SVC': SVC()
}

# Parametros de los modelos para el Test
params = { 
    'ExtraTreesClassifier': { 
        'n_estimators': [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
    },
    'GradientBoostingClassifier': { 
        'n_estimators': [17, 22, 27, 32], 
        'learning_rate': [0.3, 0.5, 0.8, 1.0],
    },
    'SVC': {
        'kernel': ['rbf'], 
        'C': [1, 3, 5, 7, 9], 
        'gamma': [0.1, 0.01, 0.001, 0.0001],
    }
}       

params = {
	'k-NN': {
		'beacons_' : ['Be01', 'Be02', 'Be03', 'Be04', 'Be05']
		'freq_' : ['Tx_0x01', 'Tx_0x02', 'Tx_0x03', 'Tx_0x04', 'Tx_0x05', 'Tx_0x06', 'Tx_0x07']
	}
}
potencia = list(range(1,8))
'Tx_0x0'+str(potencia)

helper = EstimatorSelection(models, params)
helper.fit(x_train, y_train, scoring="accuracy", n_jobs=4, tournament_size=3,
	population_size=20, gene_mutation_prob=0.20, gene_crossover_prob=0.6, generations_number=10)