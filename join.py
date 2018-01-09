"""
	Unión de datasets
"""
import pandas as pd
names_ = ['Sector', 'Be01', 'Be02', 'Be03', 'Be04', 'Be05']
for potencia in range(1,8):
	txsum = pd.DataFrame()
	for i in range(1,16):
		txsum = txsum.append(pd.read_csv("Raspberri/Tx_0x0"+str(potencia)+"/train"+str(i)+".csv", names=names_))
	txsum = txsum[['Be01', 'Be02', 'Be03', 'Be04', 'Be05', 'Sector']]
	txsum.to_csv('Tx_0x0'+str(potencia), sep=',', index=False)
	potencia = potencia + 1

from sklearn.utils import shuffle
columna = names_[0]
# fila
t1 = shuffle(pd.read_csv('Tx_0x01')).reset_index(drop=True)[:794]
t2 = shuffle(pd.read_csv('Tx_0x02')).reset_index(drop=True)[:794]
t3 = shuffle(pd.read_csv('Tx_0x03')).reset_index(drop=True)[:794]
t4 = shuffle(pd.read_csv('Tx_0x04')).reset_index(drop=True)[:794]
t5 = shuffle(pd.read_csv('Tx_0x05')).reset_index(drop=True)[:794]
t6 = shuffle(pd.read_csv('Tx_0x06')).reset_index(drop=True)[:794]
t7 = shuffle(pd.read_csv('Tx_0x07')).reset_index(drop=True)[:794]

dataset = t1['Be01'], t1['Be02'], t1['Be03'], t1['Be04'], t1['Be05']  e
