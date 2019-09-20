#CLasificacion HSI usando SOM + CNN INCEPTION
#Se cargan los datos reducidos dimensionalmente utilizando SOM y el toolbox de RNA de MATLAB

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from package.prepararDatos import PrepararDatos
from package.firmasEspectrales import FirmasEspectrales
from package.dataLogger import DataLogger
from keras import layers
from keras.models import Model
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np

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
#CREAR FICHERO DATA LOGGER 
logger = DataLogger(dataSet+'SOMINC')  
#FIRMAS ESPECTRALES
espectros = FirmasEspectrales(imagen, groundTruth)
firmas = espectros.promediarFirmas() # Promedios de todas las firmas espectrales
#GRAFICAR FIRMAS ESPECTRALES
espectros.graficarFirmas(firmas) # Grafica de los promedios de todas las firmas espectrales
#COMIENZAN LAS ITERACIONES
OA = 0
vectOA = np.zeros(numTest)
for i in range(0, numTest):
    #PREPARAR DATOS PARA ENTRENAMIENTO
    preparar = PrepararDatos(imagen, groundTruth, False)
    datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,30,ventana)
    datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)

    #DEFINICION RED CONVOLUCIONAL INCEPTION 
    input_tensor = Input(shape=(datosEntrenamiento.shape[1],datosEntrenamiento.shape[2],datosEntrenamiento.shape[3]))
    # Cada rama tiene el mismo estado de padding='same', lo cual es necesario para mantener todas las salidas de las ramas 
    # en el mismo tamaño. Esto posibilita la ejecución de la instrucción concatenate.
    # Rama A
    branch_a = layers.Conv2D(128, (1,1), activation='relu', padding='same')(input_tensor)
    # Rama B
    branch_b = layers.Conv2D(128, (1,1), activation='relu', padding='same')(input_tensor)
    branch_b = layers.Conv2D(128, (3,3), activation='relu', padding='same')(branch_b)
    # Rama C
    branch_c = layers.AveragePooling2D((3,3), strides=(1,1), padding='same')(input_tensor)
    branch_c = layers.Conv2D(128, (3,3), activation='relu', padding='same')(branch_c)
    # Rama D
    branch_d = layers.Conv2D(128, (1,1), activation='relu', padding='same')(input_tensor)
    branch_d = layers.Conv2D(128, (3,3), activation='relu', padding='same')(branch_d)
    branch_d = layers.Conv2D(128, (3,3), activation='relu', padding='same')(branch_d)
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
    model.save('hsiSOMandINCEPTION'+str(i)+'.h5')

#GENERAR MAPA FINAL DE CLASIFICACIÓN
datosSalida = model.predict(datosPrueba)
datosSalida = preparar.predictionToImage(datosSalida)
#GRAFICAS
graficarHsi_VS(groundTruth, datosSalida)
logger.close()