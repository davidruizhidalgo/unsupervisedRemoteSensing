#CLasificacion HSI usando SOM + CNN INCEPTION
#Se cargan los datos reducidos dimensionalmente utilizando SOM y el toolbox de RNA de MATLAB

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from package.prepararDatos import PrepararDatos
from package.firmasEspectrales import FirmasEspectrales
from keras import layers
from keras import models
from keras import regularizers
from io import open

def loadSomData(name_data):
    dicData = {'IndianPines' : ['C:/Users/david/Documents/dataSets/DatosSOM7/dataSOM_1.mat', 'dataSOM', 'C:/Users/david/Documents/dataSets/Indian_pines_gt.mat', 'indian_pines_gt'],
                'Salinas' : ['C:/Users/david/Documents/dataSets/DatosSOM7/dataSOM_2.mat', 'dataSOM', 'C:/Users/david/Documents/dataSets/Salinas_gt.mat', 'salinas_gt'],
                'PaviaU' : ['C:/Users/david/Documents/dataSets/DatosSOM7/dataSOM_3.mat', 'dataSOM', 'C:/Users/david/Documents/dataSets/PaviaU_gt.mat', 'paviaU_gt'], }
        #CARGAR CUBO DE DATOS
    mat = sio.loadmat(dicData[name_data][0]) # Cargar archivo .mat
    data = np.array(mat[dicData[name_data][1]]) # Convertir a numpy array
    data = data.T   # Transponer para ajustar los ejes coordenados      
    data_t = np.zeros( (data.shape[0],data.shape[2],data.shape[1]) )
    for i in range(data.shape[0]): # Transponer cada canal para ajustar los ejes coordenados
        data_t[i] = data[i].T 
    #NORMALIZAR DATOS DE ENTRADA
    mean = data_t.mean(axis=0)
    data_t -= mean
    std = data_t.std(axis=0)
    data_t /= std
    imagen = data_t.copy()   #IMAGEN DE ENTRADA SOM
    #CARGAR GROUND TRUTH
    mat = sio.loadmat(dicData[name_data][2]) # Cargar archivo Ground Truth .mat
    data = np.array(mat[dicData[name_data][3]]) # Convertir Ground Truth a numpy array
    groundTruth = data.copy()         #GROUND TRUTH
    return imagen, groundTruth

def graficarHsi_VS(img_1, img_2):
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(img_1)
    plt.subplot(1,2,2)
    plt.imshow(img_2)
    plt.show()
###############################################################################################
#Programa principal############################################################################
###############################################################################################
#CARGAR DATOS
numTest = 1
dataSet = 'IndianPines'
ventana = 9 #VENTANA 2D de PROCESAMIENTO
imagen, groundTruth = loadSomData(dataSet)

nlogg = 'logger_'+dataSet+'SOMCNN.txt'
fichero = open(nlogg,'w')  
fichero.write('Datos SOM + INCEPTION')   
#FIRMAS ESPECTRALES
espectros = FirmasEspectrales(imagen, groundTruth)
firmas = espectros.promediarFirmas() # Promedios de todas las firmas espectrales
#GRAFICAR FIRMAS ESPECTRALES
espectros.graficarFirmas(firmas) # Grafica de los promedios de todas las firmas espectrales
#COMIENZAN LAS ITERACIONES
for i in range(0, numTest):
    #PREPARAR DATOS PARA ENTRENAMIENTO
    preparar = PrepararDatos(imagen, groundTruth, False)
    datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,30,ventana)
    datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)

    #DEFINICION RED CONVOLUCIONAL
    model = models.Sequential()
    model.add(layers.Conv2D(128, (5, 5), kernel_regularizer=regularizers.l2(0.001),activation='relu', input_shape=(datosEntrenamiento.shape[1],datosEntrenamiento.shape[2],datosEntrenamiento.shape[3])))
    model.add(layers.Conv2D(128, (3, 3), kernel_regularizer=regularizers.l2(0.001),activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), kernel_regularizer=regularizers.l2(0.001),activation='relu'))

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
    history = model.fit(datosEntrenamiento,etiquetasEntrenamiento,epochs=25,batch_size=512,validation_data=(datosValidacion, etiquetasValidacion))

    #EVALUAR MODELO
    test_loss, test_acc = model.evaluate(datosPrueba, etiquetasPrueba)
    print(test_acc)

    #LOGGER DATOS DE ENTRENAMIENTO
    #CREAR DATA LOGGER
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    fichero.write('\n'+str(loss))
    fichero.write('\n'+str(val_loss))
    fichero.write('\n'+str(acc))
    fichero.write('\n'+str(val_acc))
        
    #GUARDAR MODELO DE RED CONVOLUCIONAL
    model.save('hsiSOM_CNN2D'+str(i)+'.h5')

#GENERAR MAPA FINAL DE CLASIFICACIÓN
datosSalida = model.predict(datosPrueba)
datosSalida = preparar.predictionToImage(datosSalida)
#GRAFICAS
graficarHsi_VS(groundTruth, datosSalida)
fichero.close()