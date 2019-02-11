import pandas as pd
import numpy as np

names_ = ['Be01', 'Be02', 'Be03', 'Be04', 'Be05', 'Sector']
frecuencias = []
frecuencias.append(pd.read_csv('Filtrado/LocalizationNew_Tx1.csv', names=names_))
frecuencias.append(pd.read_csv('Filtrado/LocalizationNew_Tx2.csv', names=names_))
frecuencias.append(pd.read_csv('Filtrado/LocalizationNew_Tx3.csv', names=names_))
frecuencias.append(pd.read_csv('Filtrado/LocalizationNew_Tx4.csv', names=names_))
frecuencias.append(pd.read_csv('Filtrado/LocalizationNew_Tx5.csv', names=names_))
frecuencias.append(pd.read_csv('Filtrado/LocalizationNew_Tx6.csv', names=names_))
#frecuencias.append(pd.read_csv('Filtrado/LocalizationNew_Tx7.csv'))

salida = [pd.DataFrame() for i in range(6)]
for sec in range(1,16):
	corte = min([frecuencias[i][frecuencias[i]['Sector']==sec].shape[0] for i in range(6)])
	for tx in range(6):
		salida[tx] = salida[tx].append(frecuencias[tx][frecuencias[tx]['Sector']==sec][1:corte])

for i in range(6):
	salida[i].to_csv('Tx_0x0'+str(i+1), sep=',', index=False) 