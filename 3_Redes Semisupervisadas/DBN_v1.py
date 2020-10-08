# Deep Belief Network for Hyperspectral Image Classification
import warnings
warnings.filterwarnings('ignore')
from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from package.dataLogger import DataLogger

import matplotlib.pyplot as plt
import numpy as np
import os 

from keras.layers import Dense
from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import cohen_kappa_score
from keras import backend as K 


#CARGAR IMAGEN HSI Y GROUND TRUTH
numTest = 10                  # Number of iterations
dataSet = 'KSC'      # Data set
test =  'DBN'                # Deep Belief Network

ventana = 9                  #VENTANA 2D de PROCESAMIENTO
data = CargarHsi(dataSet)
imagen = data.imagen
groundTruth = data.groundTruth

#CREAR FICHERO DATA LOGGER 
logger = DataLogger(dataSet+'_TEST',test) 

#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis()
imagenFE = pca.pca_calculate(imagen, varianza=0.95)
imagenFE = (imagenFE-np.amin(imagenFE))/(np.amax(imagenFE)-np.amin(imagenFE)) # Normalizar datos
print(imagenFE.shape)

for i in range(0, numTest):
    #PREPARAR DATOS PARA ENTRENAMIENTO
    preparar = PrepararDatos(imagenFE, groundTruth, False)
    datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,20,ventana)
    datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)
    #REORGANIZAR DATOS DE ENTRADA
    datosEntrenamiento = datosEntrenamiento.reshape(datosEntrenamiento.shape[0], datosEntrenamiento.shape[1]*datosEntrenamiento.shape[2]*datosEntrenamiento.shape[3])
    datosValidacion = datosValidacion.reshape(datosValidacion.shape[0], datosValidacion.shape[1]*datosValidacion.shape[2]*datosValidacion.shape[3])
    datosPrueba = datosPrueba.reshape(datosPrueba.shape[0], datosPrueba.shape[1]*datosPrueba.shape[2]*datosPrueba.shape[3])
    ## Deep Belief Network with greedy layer wise pretraining
    layer_1 = BernoulliRBM(n_components = 512, n_iter = 50,learning_rate = 0.01,  verbose = True, random_state=0)
    layer_1.fit(datosEntrenamiento)
    features = layer_1.transform(datosEntrenamiento)
    layer_2 = BernoulliRBM(n_components = 256, n_iter = 50,learning_rate = 0.01,  verbose = True, random_state=0)
    layer_2.fit(features)
    features = layer_2.transform(features)
    layer_3 = BernoulliRBM(n_components = 256, n_iter = 50,learning_rate = 0.01,  verbose = True, random_state=0)
    layer_3.fit(features)
    # Capturar pesos y bias de cada una de las capas del DBN
    w1 = np.transpose(layer_1.components_)
    b1 = layer_1.intercept_hidden_
    w2 = np.transpose(layer_2.components_)
    b2 = layer_2.intercept_hidden_
    w3 = np.transpose(layer_3.components_)
    b3 = layer_3.intercept_hidden_

    ##################################CLASIFICADOR CAPA DE SALIDA###############################################
    classifier = models.Sequential()
    classifier.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_dim=datosEntrenamiento.shape[1], name='rbm1'))
    classifier.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='rbm2'))
    classifier.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001), name='rbm3'))
    classifier.add(layers.Dropout(0.5))
    classifier.add(Dense(64, activation='relu',  kernel_regularizer=regularizers.l2(0.001), name='fullconected'))
    classifier.add(Dense(groundTruth.max()+1, activation = 'softmax',  name='output'))
    #Copiar pesos de las capas de la DBN
    layerDN1 = classifier.get_layer('rbm1')
    layerDN1.set_weights([w1,b1])
    layerDN2 = classifier.get_layer('rbm2')
    layerDN2.set_weights([w2,b2])
    layerDN3 = classifier.get_layer('rbm3')
    layerDN3.set_weights([w3,b3])
    
    layerDN1.trainable = True
    layerDN2.trainable = True
    layerDN3.trainable = True

    print(classifier.summary())
    classifier.compile(optimizer=optimizers.Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])   
    classifier.fit(datosEntrenamiento, etiquetasEntrenamiento, epochs=400, batch_size=64, shuffle=True, validation_data=(datosValidacion, etiquetasValidacion))
    #GUARDAR CLASIFICADOR ENTRENADO
    classifier.save(os.path.join(logger.path,'C_'+test+str(i)+'.h5'))
    #VALIDACIÓN DE LA RED ENTRENADA
    test_loss, OA = classifier.evaluate(datosPrueba, etiquetasPrueba)
    print('LOGISTIC OA:'+str(OA))
    #GENERACION AA - Average Accuracy 
    AA = np.zeros(groundTruth.max()+1)
    for j in range(1,groundTruth.max()+1):                                     #QUITAR 1 para incluir datos del fondo
        datosClase, etiquetasClase = preparar.extraerDatosClase2D(ventana,j)   #MUESTRAS DE UNA CLASE                 
        datosClase = datosClase.reshape(datosClase.shape[0], datosClase.shape[1]*datosClase.shape[2]*datosClase.shape[3])
        test_loss, AA[j] = classifier.evaluate(datosClase, etiquetasClase)        #EVALUAR MODELO PARA LA CLASE
    #GENERACION Kappa Coefficient
    datosSalida = classifier.predict(datosPrueba)
    datosSalida = datosSalida.argmax(axis=1)
    #Guardar los datos de salida
    etiquetasPrueba = etiquetasPrueba.argmax(axis=1)
    kappa = cohen_kappa_score(etiquetasPrueba, datosSalida)
    #LOGGER DATOS DE VALIDACIÓN
    logger.savedataPerformance(OA, AA, kappa)

#########################GENERAR MAPA FINAL DE CLASIFICACIÓN########################
imagenSalida = preparar.predictionToImage(datosSalida)
data.graficarHsi_VS(groundTruth, imagenSalida)
K.clear_session()
logger.close()