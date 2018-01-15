estimator = KNeighborsClassifier()
scoring = "accuracy"
cv = 4
refit = True
verbose = False
population_size = 40
gene_mutation_prob = 0.2
gene_crossover_prob = 0.5
tournament_size = 3
generations_number = 10
#gene_type = gene_type
n_jobs = 4
uniform = True
error_score = 'raise'
fit_params = {}
# Parámetros adicionales
_individual_evals = {}
all_history_ = None
all_logbooks_ = None
_cv_results = None
best_score_ = None
best_params_ = None
scorer_ = None
score_cache = {}







model = KNeighborsClassifier()
model.fit(x_train, y_train)

def _individual_to_params(individual, frecuencias):
	"""[Set de parámetro según individuo]
	Arguments:
		individual {[creator.Individual]} -- [individuo]
		name_values {[list]} -- [lista de parámetros, params_data]
	Returns:
		[diccionario] -- [parámetros del individuo]
	"""
	return dict((name, values[gene]) for gene, (name, values) in zip(individual, name_values))


def _evalFunction(individual, frecuencias, scorer="accuracy", cv, uniform, fit_params,
				verbose=0, error_score='raise', score_cache={}):
	"""[Evaluación del modelo]
	Arguments:
		individual {[creator.Individual]} -- [Individuo]
		frecuencias {[list]} -- [lista de dataframes]
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
	scoring = "accuracy"
	num_folds = 4
	X,y
	maxints = [5]*5
	model = KNeighborsClassifier()

	X, y = _individual_to_params(individual, name_values)
	score = 0
	n_test = 0
	paramkey = str(individual)
	if paramkey in score_cache:
		score = score_cache[paramkey]
	else:
		kfold = KFold(n_splits=num_folds)
		cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
		score_cache[paramkey] = np.mean(cv_results)
	return (score,)




	
def fit(frecuencias):# frecuencias
	best_estimator_ = None
	best_mem_score_ = float("-inf")
	best_mem_params_ = None
	#_check_param_grid(params)
	_fit(frecuencias, params)
	if refit:
		best_estimator_ = clone(estimator)
		best_estimator_.set_params(**best_mem_params_)
		if fit_params is not None:
			best_estimator_.fit(frecuencias, **fit_params)
		else:
			best_estimator_.fit(frecuencias)

def _fit(self, frecuencias, parameter_dict):
		self._cv_results = None  # Indicador de necesidad de actualización
		self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)
		#n_samples = _num_samples(X)
		# verificar longitudes x,y 
		#if _num_samples(y) != n_samples:
		#	raise ValueError('Target [y], data [X] dont agree')
		#cv = check_cv(self.cv, y=y, classifier=is_classifier(self.estimator))
		toolbox = base.Toolbox()
		# name_values = lista de parametros, gene_type = [1:categorico; 2:numérico], maxints = size(parametros)
		name_values, self.gene_type, maxints = _get_param_types_maxint(parameter_dict)
		maxints = [5]*5
		#if self.verbose:
		#	print("Tipos: %s, rangos: %s" % (self.gene_type, maxints))
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
						name_values=name_values, frecuencias,
						scorer=self.scorer_, cv=cv, uniform=self.uniform, verbose=self.verbose,
						error_score=self.error_score, fit_params=self.fit_params,
						score_cache=self.score_cache)
		# registro de función Cruce
		toolbox.register("mate", _cxIndividual, prob_cruce=self.gene_crossover_prob)
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
		pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=self.gene_crossover_prob, 
										mutpb=self.gene_mutation_prob,
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
		#if self.verbose:
		#	print("Best individual is: %s\nwith fitness: %s" % (
		#		current_best_params_, current_best_score_))
		if current_best_score_ > self.best_mem_score_:
			self.best_mem_score_ = current_best_score_
			self.best_mem_params_ = current_best_params_
		# fin paralelización, close pool
		pool.close()
		pool.join()
		self.best_score_ = current_best_score_
		self.best_params_ = current_best_params_
