# Numero de K-PCA y red CNN 2D
from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from sklearn.metrics import cohen_kappa_score
from keras import layers
from keras import models
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from keras import backend as K 

#CARGAR IMAGEN HSI Y GROUND TRUTH
numTest = 10
dataSet = 'Urban'
ventana = 9 #VENTANA 2D de PROCESAMIENTO
data = CargarHsi(dataSet)
imagen = data.imagen
groundTruth = data.groundTruth

#ANALISIS DE COMPONENTES PRINCIPALES PCA o KPCA
pca = princiapalComponentAnalysis()

vectOA = np.zeros(numTest)
vectAA = np.zeros(numTest)
vectkappa = np.zeros(numTest)
for i in tqdm(range(0, numTest)):
    #PREPARAR DATOS PARA ENTRENAMIENTO
    imagenPCA = pca.pca_calculate(imagen, componentes= 2*i+2)
    print(imagenPCA.shape)
    print('DIMENSIONAL REDUCTION DONE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    preparar = PrepararDatos(imagenPCA, groundTruth, False)
    datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,30,ventana)
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

    #EVALUAR MODELO DE RED CONVOLUCIONAL
    #GENERACION OA - Overall Accuracy 
    datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)   #TOTAL MUESTRAS                      
    test_loss, OA = model.evaluate(datosPrueba, etiquetasPrueba)            #EVALUAR MODELO
    vectOA[i] = OA                                                          #Almacena el OA de cada prueba
    #GENERACION AA - Average Accuracy 
    AA = 0 
    ClassAA = np.zeros(groundTruth.max()+1)
    for j in range(1,groundTruth.max()+1):                                   #QUITAR 1 para incluir datos del fondo
        datosClase, etiquetasClase = preparar.extraerDatosClase2D(ventana,j) #MUESTRAS DE UNA CLASE                 
        test_loss, ClassAA[j] = model.evaluate(datosClase, etiquetasClase)   #EVALUAR MODELO PARA LA CLASE
        AA += ClassAA[j]
    AA /= groundTruth.max()                                                  #SUMAR 1 para incluir datos del fondo
    vectAA[i] = AA                                                           #Almacena el AA de cada prueba
    
    #GENERACION Kappa Coefficient
    datosSalida = model.predict(datosPrueba)
    datosSalida = datosSalida.argmax(axis=1)
    etiquetasPrueba = etiquetasPrueba.argmax(axis=1)
    kappa = cohen_kappa_score(etiquetasPrueba, datosSalida)
    vectkappa[i] = kappa

    print('Iteration '+ str(i+1)+' DONE !!!!!!')                          #Overall Accuracy 

#SALIDA INDICES DE DESEMPEÑO
print('OA = '+ str(vectOA))                          #Overall Accuracy 
print('AA = '+ str(vectAA))                          #Average Accuracy 
print('kappa = '+ str(vectkappa))                    #Kappa Coefficient
K.clear_session()