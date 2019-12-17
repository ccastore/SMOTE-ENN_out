#------------------------import the package----------------------------------------------

import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
from numpy import genfromtxt
import shutil
from heapq import nsmallest
from sklearn.neighbors import NearestNeighbors
from collections import Counter
import sklearn
import tensorflow as tf
from tensorflow import keras
from sklearn.utils import shuffle
from keras.models import load_model
import h5py

#----------------------------------Method------------------------------------------------
def ENN_out(Base,Method,Model,Weights,K,Random_state):
	directorio=str("/media/carlos/18087B7FDABD7C7F/Archivos/"+Base) 
	carga=np.loadtxt(str(directorio+"/"+str(Base)+"_entrenamiento"+str(Method)+".txt"))
	datos=carga[:,0:carga.shape[1]-1]
	clases=carga[:,carga.shape[1]-1]

	#Se aplica SMOTE a los datos
	x=datos
	y=clases
	x_res,y_res = shuffle(x, y, random_state=Random_state)

	#Se carga modelo de red 

	modelo = str(directorio+str(Model))
	pesos = str(directorio+str(Weights))
	model = tf.keras.models.load_model(modelo)
	model.load_weights(pesos)

	#Se evaluan los datos en el modelo
	snn_pred = model.predict(x_res, batch_size=100)

	datos=open(str(directorio+"/"+Base+"_entrenamiento"+str(Method)+"-ENNout.txt"),'w')

	x1=x_res
	y1=y_res

	Kvecinos=K
	Kvecinos=Kvecinos+1
	samples = snn_pred
	a=0
	b=0

	neigh = NearestNeighbors(Kvecinos,1,metric='euclidean')
	neigh.fit(samples)
	for j in tqdm(range (samples.shape[0])):
	  vecinos=neigh.kneighbors([samples[j][:]], Kvecinos, return_distance=False)
	  contador=0
	  for i in range (Kvecinos):
	    if y1[vecinos[0][i]] == y1[j]:
	      contador=contador+1

	  if contador >= ((Kvecinos/2)+1):
	    a=a+1
	    for k in range (int(x1.shape[1])):
	      datos.write(str(x1[j][k])+" ")
	    datos.write(str(y1[j]))
	    datos.write(os.linesep)

	  else:
	    b=b+1
	datos.close()
#------------------------------------------------------------------------------------------

ENN_out("KSC","_SMOTE","/modeloSMOTE.h5","/pesosSMOTE.h5",5,42)
