### frecuencias
configuracion = set_config() #dict ['name']=configuracion
estimators = set_models() #list (...(name, model))

num_splits = 10
kfold = KFold(n_splits=num_splits, shuffle=False)

for name, model in estimators:
    # name, model     # nombre y modelo
    grilla_parametros = get_params(name)   # parametros
    grilla_parametros = dict((key.split(name+'__')[-1], value) for (key,value) in grilla_parametros.items())
    X, y = _createDataset(frecuencias, configuracion[name])
    grid_search_t = GridSearchCV(model, grilla_parametros, n_jobs = -1, verbose=1, cv = kfold)
    print('GridSearch for ',name)
    grid_search_t.fit(X, y)
    reporte = {}
    df = pd.DataFrame(grid_search_t.grid_scores_[:3])
    df['Model'] = name
    reporte['modelo'] = name
    reporte['best_score'] = round(grid_search_t.best_score_, 3)
    reporte['fits'] = len(grid_search_t.grid_scores_)*num_splits
    reporte['parametros'] = 
