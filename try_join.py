def _createDataset(frecuencias, values):
    # crear dataset
    names_ = salida[0].columns.values
    # reestructuracion
    dataset = pd.DataFrame()
    #salida = [pd.DataFrame() for i in range(6)]
    for sec in range(1,16):
        corte = min([frecuencias[i][frecuencias[i]['Sector']==sec].shape[0] for i in values])
        for tx in range(5):
            #salida[tx] = salida[tx].append(frecuencias[values[tx]][frecuencias[values[tx]]['Sector']==sec][1:corte])
            dataset[names_[tx]] = frecuencias[int(values[tx])][frecuencias[int(values[tx])]['Sector']==sec][1:corte][names_[tx]]
        dataset[names_[5]] = sec
    #salida = salida.append(frecuencias[tx][frecuencias[tx]['Sector']==sec][1:corte])
    salida[0] = shuffle(salida[0], random_state=seed).reset_index(drop=True)
    salida[1] = shuffle(salida[1], random_state=seed).reset_index(drop=True)
    salida[2] = shuffle(salida[2], random_state=seed).reset_index(drop=True)
    salida[3] = shuffle(salida[3], random_state=seed).reset_index(drop=True)
    salida[4] = shuffle(salida[4], random_state=seed).reset_index(drop=True)
    salida[5] = shuffle(salida[5], random_state=seed).reset_index(drop=True)
    
    dataset = pd.DataFrame()
    dataset[names_[0]] = salida[int(values[0])][names_[0]]
    dataset[names_[1]] = salida[int(values[1])][names_[1]]
    dataset[names_[2]] = salida[int(values[2])][names_[2]]
    dataset[names_[3]] = salida[int(values[3])][names_[3]]
    dataset[names_[4]] = salida[int(values[4])][names_[4]]
    dataset[names_[5]] = salida[0][names_[5]]
    dataset = dataset.dropna(how='any')
    # separación de data en X,y 
    y = dataset[names_[5]]
    del dataset[names_[5]]
    X = dataset
    return X,y