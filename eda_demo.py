#!/bin/env python

from scipy import *
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np

# The problem to optimize
def getAccuracy( frecuencias, individual, estimator, score_cache ):
	X,y = _createDataset(frecuencias, individual)
	score = 0
	scorer = "accuracy"
	paramkey = str(individual)
	if paramkey in score_cache:
		score = score_cache[paramkey]
	else:
		kfold = KFold(n_splits=10, shuffle=False)
		cv_results = cross_val_score(estimator, X, y, cv=kfold, scoring=scorer)
		score = np.mean(cv_results)
		score_cache[paramkey] = score
	return score

def _createDataset(frecuencias, values):
    names_ = frecuencias[0].columns.values
    dataset = pd.DataFrame()
    dataset[names_[0]] = frecuencias[int(values[0])][names_[0]]
    dataset[names_[1]] = frecuencias[int(values[1])][names_[1]]
    dataset[names_[2]] = frecuencias[int(values[2])][names_[2]]
    dataset[names_[3]] = frecuencias[int(values[3])][names_[3]]
    dataset[names_[4]] = frecuencias[int(values[4])][names_[4]]
    dataset[names_[5]] = frecuencias[0][names_[5]]
    # separación de data en X,y 
    y = dataset[names_[5]]
    del dataset[names_[5]]
    X = dataset
    return X,y

class eda:
	def __init__(self, of, frecuencias, estimator):
		# Algorithm parameters
		self.iterations = 100
		self.sample_size = 60
		self.select_ratio = 0.6
		self.epsilon = 10e-6

		# class members
		self.objective_function = of
		self.dimensions = 5
		self.sample = []
		self.means = []
		self.stdevs = []	

		self.debug = True
		# aditional parameters
		self.frecuencias = frecuencias
		self.estimator = estimator
		self.score_cache = {}


	def sample_sort(self): 
		# sort rows on the last column
		self.sample = self.sample[ np.argsort( self.sample[:,-1], 0 ) ]


	def dispersion_reduction(self):
		self.sample_sort()

		# number of points to select
		nb = int( np.floor( self.sample_size * self.select_ratio ) )

		# selection
		#self.sample = self.sample[:nb]
		self.sample = self.sample[self.sample_size-nb:]

		if self.debug:
		    print ("dispersion reduction")
		    print (str(self.sample))
		    print


	def estimate_parameters( self ):
		# points sub array (without values)
		mat = self.sample[:,:self.dimensions]
		
		# row means (axis 0 in scipy)
		self.means = mean( mat, 0 )
		
		# row standard deviation
		self.stdevs = std( mat, 0 )

		if self.debug:
		    print ("estimate parameters")
		    print ("\tmean=" +str(self.means))
		    print ("\tstd-dev=" + str(self.stdevs))
		    print


	def draw_sample(self):
		# for each variable to optimize
		for i in range(self.dimensions):
			# if the dispersion is null
			if self.stdevs[i] == 0.0:
				# set it to a minimal value
				self.stdevs[i] = self.epsilon
		
		# empty sample
		self.sample = np.zeros( (self.sample_size, self.dimensions+1) )
		
		# for each point
		for i in range( self.sample_size ):
			# draw in random normal
			p = np.random.normal( self.means, self.stdevs )
			p = np.array([0 if i<0 else (5 if i>5 else i) for i in p])
			# put it into the sample
			self.sample[i][:self.dimensions] = np.round(p)%(self.dimensions+1)

		if self.debug:
		    print ("draw sample")
		    print (self.sample)
		    print


	def evaluate(self):
		# for each point
		for i in range( self.sample_size ):
			d = self.dimensions
			# call the objective function
			#   the third element is the result of the objective function call
			#   taking the first two elements as variables
			r = self.objective_function( self.frecuencias, self.sample[i][:d], self.estimator, self.score_cache )
			self.sample[i][-1] = r

		if self.debug:
		    print ("evaluate")
		    print (self.sample)
		    print


	def run(self):
		# uniform initialization
		self.sample = np.random.rand( self.sample_size, self.dimensions+1 )
		# cosmetic
		#self.sample = self.sample * 200 - 100
		top_freq = 6
		self.sample = np.floor(np.random.rand(self.sample_size, self.dimensions +1)*top_freq)
		
		if self.debug:
		    print ("initialization")
		    print (self.sample)
		    print

		self.evaluate()

		
		i = 0
		while i < self.iterations:
			if self.debug:
			    print ("iteration",i)
			    print

			i += 1
			self.dispersion_reduction()
			self.estimate_parameters()
			self.draw_sample()
			self.evaluate()


		# sort the final sample
		self.sample_sort()
		# output the optimum
		ranking = self.sample_size
		print ("#[ Configuración ]\t Accuracy")
		for i in range(ranking):
			linea = str(self.sample[-i-1][:-1]) + "\t" +str(self.sample[-i-1][-1])
			print(linea)

		


if __name__=="__main__":
	seed = 7
	frecuencias = []
	frecuencias.append(pd.read_csv('Tx_0x01'))
	frecuencias.append(pd.read_csv('Tx_0x02'))
	frecuencias.append(pd.read_csv('Tx_0x03'))
	frecuencias.append(pd.read_csv('Tx_0x04'))
	frecuencias.append(pd.read_csv('Tx_0x05'))
	frecuencias.append(pd.read_csv('Tx_0x06'))
	frecuencias.append(pd.read_csv('Tx_0x07'))
	frecuencias[0] = shuffle(frecuencias[0], random_state=seed).reset_index(drop=True)
	frecuencias[1] = shuffle(frecuencias[1], random_state=seed).reset_index(drop=True)
	frecuencias[2] = shuffle(frecuencias[2], random_state=seed).reset_index(drop=True)
	frecuencias[3] = shuffle(frecuencias[3], random_state=seed).reset_index(drop=True)
	frecuencias[4] = shuffle(frecuencias[4], random_state=seed).reset_index(drop=True)
	frecuencias[5] = shuffle(frecuencias[5], random_state=seed).reset_index(drop=True)
	frecuencias[6] = shuffle(frecuencias[6], random_state=seed).reset_index(drop=True)
	estimator = KNeighborsClassifier(n_jobs=-1)
	a = eda( getAccuracy, frecuencias, estimator )
	a.run()



