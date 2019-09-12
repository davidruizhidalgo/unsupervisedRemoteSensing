#ENTRENAMIENTO DE RED CONVOLUCIONAL INCPETION 3D - CLASIFICACION HSI 
#Se utiliza PCA para reduccion dimensional. A la red inception 3D se introduce un tensor 5D de la imagen original para la
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
from keras.models import Model
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD
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

    #DEFINICION RED CONVOLUCIONAL INCEPTION 
    input_tensor = Input(shape=(datosEntrenamiento.shape[1],datosEntrenamiento.shape[2],datosEntrenamiento.shape[3],1))
    # Cada rama tiene el mismo estado de padding='same', lo cual es necesario para mantener todas las salidas de las ramas 
    # en el mismo tamaño. Esto posibilita la ejecución de la instrucción concatenate.
    # Rama A
    branch_a = layers.Conv3D(64, (1,1,32), data_format='channels_last', activation='relu', padding='same')(input_tensor)
    # Rama B
    branch_b = layers.Conv3D(64, (1,1,32), data_format='channels_last', activation='relu', padding='same')(input_tensor)
    branch_b = layers.Conv3D(64, (3,3,32), data_format='channels_last', activation='relu', padding='same')(branch_b)
    # Rama C
    branch_c = layers.AveragePooling3D((3,3,32), data_format='channels_last', strides=(1,1,1), padding='same')(input_tensor)
    branch_c = layers.Conv3D(64, (3,3,32), data_format='channels_last', activation='relu', padding='same')(branch_c)
    # Rama D
    branch_d = layers.Conv3D(64, (1,1,32), data_format='channels_last', activation='relu', padding='same')(input_tensor)
    branch_d = layers.Conv3D(64, (3,3,32), data_format='channels_last', activation='relu', padding='same')(branch_d)
    branch_d = layers.Conv3D(64, (3,3,32), data_format='channels_last', activation='relu', padding='same')(branch_d)
    # Se concatenan todas las rama  para tener un solo modelo en output
    output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)
    # Se añade como capa final de salida un clasificador tipo Multinomial logistic regression
    output = Flatten()(output)
    out    = Dense(groundTruth.max()+1, activation='softmax')(output)
    # Se define el modelo total de la red 
    model = Model(inputs = input_tensor, outputs = out)
    #print(model.summary())
    #ENTRENAMIENTO DE LA RED CONVOLUCIONAL INCEPTION
    epochs = 25
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    history = model.fit(datosEntrenamiento,etiquetasEntrenamiento,epochs=epochs,batch_size=256,validation_data=(datosValidacion, etiquetasValidacion))

    #EVALUAR MODELO
    test_loss, test_acc = model.evaluate(datosPrueba, etiquetasPrueba)
    vectOA[i] = test_acc
    OA = OA+test_acc
    #LOGGER DATOS DE ENTRENAMIENTO
    logger.savedataTrain(history)
    #GUARDAR MODELO DE RED CONVOLUCIONAL
    model.save('hsiINCEPTION3D'+str(i)+'.h5')

#GENERAR MAPA FINAL DE CLASIFICACIÓN
print('dataOA = '+ str(vectOA)) 
print('OA = '+ str(OA/numTest)) 
datosSalida = model.predict(datosPrueba)
datosSalida = preparar.predictionToImage(datosSalida)
#GRAFICAS
data.graficarHsi_VS(groundTruth, datosSalida)
logger.close()
