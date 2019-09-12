#ENTRENAMIENTO DE RED CONVOLUCIONAL 3D - CLASIFICACION HSI 
#Se utiliza PCA para reduccion dimensional. A la red convolucional 3D se introduce un tensor 5D de la imagen original para la
#generacion de caracteristicas espectrales-espaciales a partir de la convolucion. Se utiliza como capa de salida un clasificador
# tipo Multinomial logistic regression. Todas las capas utilizan entrenamiento supervisado. 

###INICIO FORZAR EJECUCION EN LA CPU
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'    
###FIN FORZAR EJECUCION EN LA CPU

from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from package.dataLogger import DataLogger
from keras import layers
from keras import models
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np

#CARGAR IMAGEN HSI Y GROUND TRUTH
numTest = 10
dataSet = 'IndianPines'
ventana = 9 #VENTANA 2D de PROCESAMIENTO
data = CargarHsi(dataSet)
imagen = data.imagen
groundTruth = data.groundTruth

#CREAR FICHERO DATA LOGGER 
logger = DataLogger(dataSet)    

#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis()
imagenPCA = pca.pca_calculate(imagen, varianza=0.95)
#imagenPCA = pca.pca_calculate(imagen, componentes=4)
print(imagenPCA.shape)
OA = 0
vectOA = np.zeros(numTest)
for i in range(0, numTest):
    #PREPARAR DATOS PARA ENTRENAMIENTO
    preparar = PrepararDatos(imagenPCA, groundTruth, False)
    datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos3D(50,30,ventana)
    datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba3D(ventana)

    #DEFINICION RED CONVOLUCIONAL
    model = models.Sequential()
    model.add(layers.Conv3D(48, (5,5,32), data_format='channels_last', padding='same', kernel_regularizer=regularizers.l2(0.001),activation='relu', input_shape=(datosEntrenamiento.shape[1],datosEntrenamiento.shape[2],datosEntrenamiento.shape[3],1)))
    model.add(layers.MaxPooling3D((2,2,2), data_format='channels_last', strides=(2,2,2), padding='valid'))
    model.add(layers.Conv3D(96, (5,5,32), data_format='channels_last', padding='same', kernel_regularizer=regularizers.l2(0.001),activation='relu'))
    model.add(layers.MaxPooling3D((2,2,2), data_format='channels_last', strides=(1,1,1), padding='valid'))
    model.add(layers.Conv3D(96, (5,5,32), data_format='channels_last', padding='same', kernel_regularizer=regularizers.l2(0.001),activation='relu'))
    model.add(layers.MaxPooling3D((2,2,2), data_format='channels_last', strides=(1,1,1), padding='valid'))
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
    history = model.fit(datosEntrenamiento,etiquetasEntrenamiento,epochs=25,batch_size=128,validation_data=(datosValidacion, etiquetasValidacion))

    #EVALUAR MODELO
    test_loss, test_acc = model.evaluate(datosPrueba, etiquetasPrueba)
    vectOA[i] = test_acc
    OA = OA+test_acc
    #LOGGER DATOS DE ENTRENAMIENTO
    logger.savedataTrain(history)
    #GUARDAR MODELO DE RED CONVOLUCIONAL
    model.save('hsiCNN3D'+str(i)+'.h5')

#GENERAR MAPA FINAL DE CLASIFICACIÓN
print('dataOA = '+ str(vectOA)) 
print('OA = '+ str(OA/numTest)) 
datosSalida = model.predict(datosPrueba)
datosSalida = preparar.predictionToImage(datosSalida)
#GRAFICAS
data.graficarHsi_VS(groundTruth, datosSalida)
logger.close()
