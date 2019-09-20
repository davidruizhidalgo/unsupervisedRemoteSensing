#Script utilizado para el desarrollo de pruebas en el codigo. 
####### EN DESARROLLO ##########################
# 1. Desarrollar una arquitectura de red selección de caracteristicas NO SUPERVISADA - Capa de Salida FINE-TUNNING
        # CNN Autoendoder
        # Grafo de CNN autoencoders
        # Evaluar funciones de distancia en la funcion de costo
        # Evaluar estrategias de concatenacion en BSCAE

# 2. Desarrollar un esquema de data augmentation => prepararDatos.py
# 4. Revisar documentacion reciente del estado del arte.

from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from package.MorphologicalProfiles import morphologicalProfiles
from package.dataLogger import DataLogger
from keras import layers
from keras import models
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np

#CARGAR IMAGEN HSI Y GROUND TRUTH
numTest = 1
dataSet = 'IndianPines'
ventana = 9 #VENTANA 2D de PROCESAMIENTO
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
imagenEEP = mp.EEP(imagenPCA, num_levels=10)    
print(imagenEEP.shape)

OA = 0
vectOA = np.zeros(numTest)
for i in range(0, numTest):
    #PREPARAR DATOS PARA ENTRENAMIENTO
    preparar = PrepararDatos(imagenEEP, groundTruth, False)
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

#GENERAR MAPA FINAL DE CLASIFICACIÓN
print('dataOA = '+ str(vectOA)) 
print('OA = '+ str(OA/numTest)) 
datosSalida = model.predict(datosPrueba)
datosSalida = preparar.predictionToImage(datosSalida)
#GRAFICAS
data.graficarHsi_VS(groundTruth, datosSalida)
