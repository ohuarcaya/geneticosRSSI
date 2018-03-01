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
        dfcv['Model'] = name_test
        column_head = ['Model', 'mean_test_score', 'mean_fit_time', 'mean_score_time', 'values', 'params']
        dfcv = dfcv[column_head].sort_values('mean_test_score', ascending=False).head(3)
        lista_df[indice] =dfcv

    except:
        lista_reporte.append({})
        lista_df[indice] = 1
    indice = indice + 1