"""
	Unión de datasets
"""
import pandas as pd
import numpy as np

names_ = ['Sector', 'Be01', 'Be02', 'Be03', 'Be04', 'Be05']
frecuencias = [pd.DataFrame() for i in range(6)]
for sector in range(1,16):
	for potencia in range(1,7):
		frecuencias[potencia-1] = frecuencias[potencia-1].append(pd.read_csv("Raspberri/Tx_0x0"+str(potencia)+"/test"+str(sector)+".csv", names=names_))
	corte = min([frecuencias[i].shape[0] for i in range(6)])
	frecuencias = [frecuencias[potencia-1][:corte] for potencia in range(1,7)]

for i in range(1,7):
	frecuencias[i-1] = frecuencias[i-1][['Be01', 'Be02', 'Be03', 'Be04', 'Be05', 'Sector']]
	frecuencias[i-1].to_csv('Tx_0x0'+str(i), sep=',', index=False) 

"""
frecuencias[0] = frecuencias[0].reset_index(drop=True)
frecuencias[1] = frecuencias[1].reset_index(drop=True)
frecuencias[2] = frecuencias[2].reset_index(drop=True)
frecuencias[3] = frecuencias[3].reset_index(drop=True)
frecuencias[4] = frecuencias[4].reset_index(drop=True)
frecuencias[5] = frecuencias[5].reset_index(drop=True)
frecuencias[6] = frecuencias[6].reset_index(drop=True)
"""
