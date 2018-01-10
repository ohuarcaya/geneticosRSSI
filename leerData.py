"""
Extracción
"""
import pandas as pd
import numpy as np

frecuencias = []
frecuencias.append(pd.read_csv('Tx_0x01'))
frecuencias.append(pd.read_csv('Tx_0x02'))
frecuencias.append(pd.read_csv('Tx_0x03'))
frecuencias.append(pd.read_csv('Tx_0x04'))
frecuencias.append(pd.read_csv('Tx_0x05'))
frecuencias.append(pd.read_csv('Tx_0x06'))
frecuencias.append(pd.read_csv('Tx_0x07'))
frecuencias[0] = shuffle(frecuencias[0], random_state=7).reset_index(drop=True)
frecuencias[1] = shuffle(frecuencias[1], random_state=7).reset_index(drop=True)
frecuencias[2] = shuffle(frecuencias[2], random_state=7).reset_index(drop=True)
frecuencias[3] = shuffle(frecuencias[3], random_state=7).reset_index(drop=True)
frecuencias[4] = shuffle(frecuencias[4], random_state=7).reset_index(drop=True)
frecuencias[5] = shuffle(frecuencias[5], random_state=7).reset_index(drop=True)
frecuencias[6] = shuffle(frecuencias[6], random_state=7).reset_index(drop=True)

names_ = frecuencias[0].columns.values
values = np.random.randint(6,size=5)
dataset = pd.DataFrame()
dataset[names_[0]] = frecuencias[values[0]][names_[0]]
dataset[names_[1]] = frecuencias[values[1]][names_[1]]
dataset[names_[2]] = frecuencias[values[2]][names_[2]]
dataset[names_[3]] = frecuencias[values[3]][names_[3]]
dataset[names_[4]] = frecuencias[values[4]][names_[4]]
dataset[names_[5]] = frecuencias[0][names_[5]]

y = dataset[names_[5]]
del dataset[names_[5]]
X = dataset	# .drop(names_[5], axis=1, inplace=True)
#x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.25, random_state=7)