#FEATURE EXTRACTION AND CLASSIFIER
# Utiliza las caracteristica generadas por los esquemas de estracción de caracteristicas no supervisados
# implementa una capa de salida supervisada.
from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from package.MorphologicalProfiles import morphologicalProfiles
from package.dataLogger import DataLogger

import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Add
from keras.layers import UpSampling2D, Dropout, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.models import Model
from keras.optimizers import SGD
from keras import regularizers

from keras import backend as K 
from keras import initializers
from keras.utils import plot_model
from keras.models import load_model

def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

#CARGAR IMAGEN HSI Y GROUND TRUTH
numTest = 1
dataSet = 'IndianPines'
ventana = 8 #VENTANA 2D de PROCESAMIENTO
data = CargarHsi(dataSet)
imagen = data.imagen
groundTruth = data.groundTruth

#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis()
imagenPCA = pca.pca_calculate(imagen, varianza=0.95)
#imagenPCA = pca.pca_calculate(imagen, componentes=4)
print(imagenPCA.shape)

#ESTIMACIÓN DE EXTENDED EXTINTION PROFILES
mp = morphologicalProfiles()
imagenEEP = mp.EEP(imagenPCA, num_levels=6)    
print(imagenEEP.shape)

#PREPARAR DATOS PARA ENTRENAMIENTO
preparar = PrepararDatos(imagenEEP, groundTruth, False)
datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,20,ventana)
datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)

######################STACKED CONVOLUTIONAL AUTOENCODER#################################################################################################
epochs = 50 #número de iteraciones
encoder = load_model('FE_SCAE0.h5', custom_objects={'euclidean_distance_loss': euclidean_distance_loss})   
#encoder = load_model('FE_BCAE1.h5', custom_objects={'euclidean_distance_loss': euclidean_distance_loss})  
print(encoder.summary()) 

features = encoder.predict(datosEntrenamiento)
features_val = encoder.predict(datosValidacion)
print(features.shape)


##################################CLASIFICADOR CAPA DE SALIDA###############################################
input_features = Input(shape=(features.shape[1],features.shape[2],features.shape[3]))
fullconected = Flatten()(input_features)
fullconected = Dense(128, activation = 'relu')(fullconected) 
fullconected = Dense(groundTruth.max()+1, activation = 'softmax')(fullconected) 
classifier = Model(inputs = input_features, outputs = fullconected) #GENERA EL MODELO FINAL

print(classifier.summary())
classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])   
history = classifier.fit(features, etiquetasEntrenamiento, epochs=epochs, batch_size=128, shuffle=True, validation_data=(features_val, etiquetasValidacion))

###########################GRAFICAS Y SALIDAS###############################
features_out = encoder.predict(datosPrueba)
datosSalida = classifier.predict(features_out)
datosSalida = preparar.predictionToImage(datosSalida)
data.graficarHsi_VS(groundTruth, datosSalida)
data.graficar_history(history)
K.clear_session()
