# Level_EEP
# pylint: disable=E1136  # pylint/issues/3139

from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from package.MorphologicalProfiles import morphologicalProfiles
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, concatenate
from keras.layers import UpSampling2D, Dropout, Conv2DTranspose, MaxPooling2D, Add
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import regularizers
from keras import backend as K 
from keras import layers
from keras.optimizers import SGD
from keras import initializers
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
import numpy as np
import os 


#############################################################################################################################################################
###########################PROGRAMA PRINCIPAL################################################################################################################
numTest = 1
dataSet = 'IndianPines'
data = CargarHsi(dataSet)
imagen = data.imagen
groundTruth = data.groundTruth
print(imagen.shape)

###########################INICIAL FEATURE EXTRACTION########################################################################################################
#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis()
imagenFE = pca.pca_calculate(imagen, varianza=0.95)
print(imagenFE.shape)

#ESTIMACIÓN DE EXTENDED EXTINTION PROFILES
mp = morphologicalProfiles()
imagenFE = mp.EEP(imagenFE, num_levels=2)   
print(imagenFE.shape)

###################CONVOLUTIONAL NEURAL NETWORK################################################################################################################
OA = 0
vectOA = np.zeros(numTest)
for i in range(0, numTest):
    #PREPARAR DATOS PARA ENTRENAMIENTO
    preparar = PrepararDatos(imagenFE, groundTruth, False)
    datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,30,ventana)
    datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)

    #DEFINICION RED CONVOLUCIONAL
    model = models.Sequential()
    model.add(layers.Conv2D(48, (5, 5), kernel_regularizer=regularizers.l2(0.001),activation='relu', input_shape=(datosEntrenamiento.shape[1],datosEntrenamiento.shape[2],datosEntrenamiento.shape[3])))
    #model.add(layers.MaxPooling2D((2,2), data_format='channels_last', strides=(1,1), padding='same'))
    model.add(layers.Conv2D(96, (3, 3), kernel_regularizer=regularizers.l2(0.001),activation='relu'))
    #model.add(layers.MaxPooling2D((2,2), data_format='channels_last', strides=(1,1), padding='same'))
    model.add(layers.Conv2D(96, (3, 3), kernel_regularizer=regularizers.l2(0.001),activation='relu'))
    #model.add(layers.MaxPooling2D((2,2), data_format='channels_last', strides=(1,1), padding='same'))
    #CAPA FULLY CONNECTED
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
    #AÑADE UN CLASIFICADOR MLR EN EL TOPE DE LA CONVNET
    model.add(layers.Dense(groundTruth.max()+1, activation='softmax'))
    print(model.summary())

    #ENTRENAMIENTO DE LA RED CONVOLUCIONAL
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(datosEntrenamiento,etiquetasEntrenamiento,epochs=35,batch_size=512,validation_data=(datosValidacion, etiquetasValidacion))

    #EVALUAR MODELO
    test_loss, test_acc = model.evaluate(datosPrueba, etiquetasPrueba)
    vectOA[i] = test_acc
    OA = OA+test_acc
    
#MOSTRAR OVERALL ACCURACY
print('OA = '+ str(OA/numTest)) 