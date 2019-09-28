#CONVOLUTIONAL STACKED AUTOENCODER - GREEDY LAYER-WISE PRETRAINING
#Se implementa el encoder y el decoder convolucional para aprender una representación de las caracteristicas de los datos de entrada.
#El proceso de entrenamiento se realiza utilizando GREEDY LAYER-WISE PRETRAINING
#Con el encoder entrenado se implementa una capa de fine tunning para el ejuste de la ultima capa del clasificador. 
#El proceso utiliza una ventana sxs de la imagen original para la generacion de caracteristicas espaciales a partir de la convolucion. 
#Se utiliza como capa de salida un clasificador tipo Multinomial logistic regression.  

from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from package.MorphologicalProfiles import morphologicalProfiles
from package.dataLogger import DataLogger

import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.models import Model
from keras.optimizers import SGD
from keras import regularizers

#CARGAR IMAGEN HSI Y GROUND TRUTH
numTest = 1
dataSet = 'IndianPines'
ventana = 8 #VENTANA 2D de PROCESAMIENTO
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

#ESTIMACIÓN DE EXTENDED ATTRIBUTE PROFILES
#mp = morphologicalProfiles()
#imagenEAP = mp.EAP(imagenPCA, num_thresholds=6)  
#print(imagenEAP.shape)

#ESTIMACIÓN DE EXTENDED EXTINTION PROFILES
mp = morphologicalProfiles()
imagenEEP = mp.EEP(imagenPCA, num_levels=6)    
print(imagenEEP.shape)

OA = 0
vectOA = np.zeros(numTest)
for i in range(0, numTest):
    #PREPARAR DATOS PARA ENTRENAMIENTO
    preparar = PrepararDatos(imagenEEP, groundTruth, False)
    datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,20,ventana)
    datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)
    #########################################################################################################################
    ########################STACKED CONVOLUTIONAL AUTOENCODER################################################################
    layer = [datosEntrenamiento.shape[3], 128, 64, 32]
    # Layer by layer pretraining Models
    # Layer 1
    input_img = Input(shape=(datosEntrenamiento.shape[1], datosEntrenamiento.shape[2], layer[0]))  #Tamaño de la entrada
    distorted_input1 = Dropout(.1)(input_img)
    encoded1 = Conv2D(layer[1], (3, 3), activation='sigmoid', padding='same')(distorted_input1) #Tamaño de la capa 1 -> relu y sigmoid
    encoded1_bn = BatchNormalization()(encoded1)
    decoded1 = Conv2D(layer[0], (3, 3), activation='sigmoid', padding='same')(encoded1_bn) #Tamaño de la entrada

    autoencoder1 = Model(inputs = input_img, outputs = decoded1) #Autoencoder 1 Completo
    encoder1 = Model(inputs = input_img, outputs = encoded1_bn)  #Encoder 1

    # Layer 2
    encoded1_input = Input(shape = (datosEntrenamiento.shape[1], datosEntrenamiento.shape[2], layer[1]))  #Tamaño de la capa 1  
    distorted_input2 = Dropout(.2)(encoded1_input)
    encoded2 = Conv2D(layer[2], (3, 3), activation='sigmoid', padding='same')(distorted_input2) #Tamaño de la capa 2  
    encoded2_bn = BatchNormalization()(encoded2)
    decoded2 = Conv2D(layer[1], (3, 3), activation='sigmoid', padding='same')(encoded2_bn)  #Tamaño de la capa 1 

    autoencoder2 = Model(inputs = encoded1_input, outputs = decoded2) #Autoencoder 2 Completo
    encoder2 = Model(inputs = encoded1_input, outputs = encoded2_bn)  #Encoder 2

    # Layer 3 - which we won't end up fitting in the interest of time
    encoded2_input = Input(shape = (datosEntrenamiento.shape[1], datosEntrenamiento.shape[2], layer[2]))    #Tamaño de la capa 2
    distorted_input3 = Dropout(.3)(encoded2_input)
    encoded3 = Conv2D(layer[3], (3, 3), activation='sigmoid', padding='same')(distorted_input3) #Tamaño de la capa 3
    encoded3_bn = BatchNormalization()(encoded3)
    decoded3 = Conv2D(layer[2], (3, 3), activation='sigmoid', padding='same')(encoded3_bn) #Tamaño de la capa 2

    autoencoder3 = Model(inputs = encoded2_input, outputs = decoded3) #Autoencoder 3 Completo
    encoder3 = Model(inputs = encoded2_input, outputs = encoded3_bn)  #Encoder 3

    #DEEP AUTOENCODER: Se define el modelo completo de Autoencoder en el que se copiaran los resultados del pre-entrenamiento 
    encoded1_da = Conv2D(layer[1], (3, 3), activation='sigmoid', padding='same')(input_img)       #Tamaño de la capa 1
    encoded1_da_bn = BatchNormalization()(encoded1_da)                
    encoded2_da = Conv2D(layer[2], (3, 3), activation='sigmoid', padding='same')(encoded1_da_bn)  #Tamaño de la capa 2
    encoded2_da_bn = BatchNormalization()(encoded2_da)
    encoded3_da = Conv2D(layer[3], (3, 3), activation='sigmoid', padding='same')(encoded2_da_bn)  #Tamaño de la capa 3
    encoded3_da_bn = BatchNormalization()(encoded3_da)
    decoded3_da = Conv2D(layer[2], (3, 3), activation='sigmoid', padding='same')(encoded3_da_bn)  #Tamaño de la capa 2
    decoded2_da = Conv2D(layer[1], (3, 3), activation='sigmoid', padding='same')(decoded3_da)     #Tamaño de la capa 1
    decoded1_da = Conv2D(layer[0], (3, 3), activation='sigmoid', padding='same')(decoded2_da)     #Tamaño de la entrada trainX.shape[1]

    deep_autoencoder = Model(inputs = input_img, outputs = decoded1_da)
    print('Definir modelos DONE!!!!')
    ###########################GREEDY LAYER-WISE PRETRAINING##########################################################################
    #Se establecen los parametros para el entrenamiento de cada capa
    sgd1 = SGD(lr = 5, decay = 0.5, momentum = .85, nesterov = True)
    sgd2 = SGD(lr = 5, decay = 0.5, momentum = .85, nesterov = True)
    sgd3 = SGD(lr = 5, decay = 0.5, momentum = .85, nesterov = True)

    autoencoder1.compile(loss='binary_crossentropy', optimizer = sgd1, metrics=['accuracy'])
    autoencoder2.compile(loss='binary_crossentropy', optimizer = sgd2, metrics=['accuracy'])
    autoencoder3.compile(loss='binary_crossentropy', optimizer = sgd3, metrics=['accuracy'])

    encoder1.compile(loss='binary_crossentropy', optimizer = sgd1, metrics=['accuracy'])
    encoder2.compile(loss='binary_crossentropy', optimizer = sgd1, metrics=['accuracy'])
    encoder3.compile(loss='binary_crossentropy', optimizer = sgd1, metrics=['accuracy'])

    deep_autoencoder.compile(loss='binary_crossentropy', optimizer = sgd1)

    #ENTRENAMIENTO CAPA POR CAPA
    #Capa 1 -> se entrena con los datos de entrada
    autoencoder1.fit(datosEntrenamiento, datosEntrenamiento, epochs = 30, batch_size = 64, validation_data=(datosValidacion, datosValidacion), shuffle = True)
    #Se generan las salidas de la capa 1
    first_layer_code = encoder1.predict(datosEntrenamiento)
    #Capa 2 -> se entrena con los datos generados por la capa 1
    autoencoder2.fit(first_layer_code, first_layer_code, epochs = 30, batch_size = 64, validation_split = 0.20, shuffle = True)
    #Se generan las salidas de la capa 2
    second_layer_code = encoder2.predict(first_layer_code)
    #Capa 3 -> se entrena con los datos generados por la capa 2
    autoencoder3.fit(second_layer_code, second_layer_code, epochs = 30, batch_size = 64, validation_split = 0.20, shuffle = True)

    #Se asigna los pesos del entrenamiento al modelo completo del Autoencoder
    deep_autoencoder.layers[1].set_weights(autoencoder1.layers[2].get_weights()) # first dense layer
    deep_autoencoder.layers[2].set_weights(autoencoder1.layers[3].get_weights()) # first bn layer
    deep_autoencoder.layers[3].set_weights(autoencoder2.layers[2].get_weights()) # second dense layer
    deep_autoencoder.layers[4].set_weights(autoencoder2.layers[3].get_weights()) # second bn layer
    deep_autoencoder.layers[5].set_weights(autoencoder3.layers[2].get_weights()) # thrird dense layer
    deep_autoencoder.layers[6].set_weights(autoencoder3.layers[3].get_weights()) # third bn layer
    deep_autoencoder.layers[7].set_weights(autoencoder3.layers[4].get_weights()) # first decoder
    deep_autoencoder.layers[8].set_weights(autoencoder2.layers[4].get_weights()) # second decoder
    deep_autoencoder.layers[9].set_weights(autoencoder1.layers[4].get_weights()) # third decoder
    print('Greedy layer-wise Done !!!!!!!!!')
    #############################DEEP AUTOENCODER TRAINING #####################################################
    deep_autoencoder.compile(loss='binary_crossentropy', optimizer = sgd1, metrics=['accuracy'])
    deep_autoencoder.fit(datosEntrenamiento, datosEntrenamiento, epochs = 30, batch_size = 64, validation_data=(datosValidacion, datosValidacion), shuffle = True)
    print('DEEP AUTOENCODER TRAINING DONE !!!!!!!!!')
    #############################FULL CONECTED LAYER ###########################################################
    #Se crea la red con la estructura del encoder DEEP AUTOENCODER y se agrega la capa FINE TUNNING
    # Deep Encoder
    ft_encoded1 = Conv2D(layer[1], (3, 3), activation='sigmoid', padding='same')(input_img)       #Tamaño de la capa 1
    ft_encoded1__bn = BatchNormalization()(ft_encoded1)                
    ft_encoded2 = Conv2D(layer[2], (3, 3), activation='sigmoid', padding='same')(ft_encoded1__bn)  #Tamaño de la capa 2
    ft_encoded2_bn = BatchNormalization()(ft_encoded2)
    ft_encoded3 = Conv2D(layer[3], (3, 3), activation='sigmoid', padding='same')(ft_encoded2_bn)  #Tamaño de la capa 3
    ft_encoded3_bn = BatchNormalization()(ft_encoded3)
    #Fine tunning
    dense_fl = Flatten()(ft_encoded3_bn)
    dense1 = Dense(128, activation = 'relu')(dense_fl)
    dense1_drop = Dropout(.3)(dense1)
    dense2 = Dense(groundTruth.max()+1, activation = 'softmax')(dense1_drop)
    #GENERA EL MODELO FINAL
    classifier = Model(inputs = input_img, outputs = dense2)
    #COPIAR y CONGELAR LOS PESOS DE LAS CAPAS ENTRENADAS
    classifier.layers[1].set_weights(deep_autoencoder.layers[1].get_weights()) # first dense layer
    classifier.layers[2].set_weights(deep_autoencoder.layers[2].get_weights()) # first bn layer
    classifier.layers[3].set_weights(deep_autoencoder.layers[3].get_weights()) # second dense layer
    classifier.layers[4].set_weights(deep_autoencoder.layers[4].get_weights()) # second bn layer
    classifier.layers[5].set_weights(deep_autoencoder.layers[5].get_weights()) # thrird dense layer
    classifier.layers[6].set_weights(deep_autoencoder.layers[6].get_weights()) # third bn layer
    #Congelar entrenamiento del encoder
    for layer in classifier.layers[1:-4]:
        layer.trainable = False
    #############################ENTRENEMIENTO CAPA DE SALIDA ###########################################################
    sgd4 = SGD(lr = .1, decay = 0.001, momentum = .8, nesterov = True)
    classifier.compile(loss='categorical_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])
    history = classifier.fit(datosEntrenamiento, etiquetasEntrenamiento, epochs = 30, batch_size = 32, validation_data=(datosValidacion, etiquetasValidacion), shuffle = True)
    ################################## VALIDACIÓN DE LA RED ENTRENADA ##################################################
    test_loss, test_acc = classifier.evaluate(datosPrueba, etiquetasPrueba)
    vectOA[i] = test_acc
    OA = OA+test_acc
    #LOGGER DATOS DE ENTRENAMIENTO
    logger.savedataTrain(history)
    #GUARDAR MODELO DE RED CONVOLUCIONAL
    classifier.save('SCAE_GLP'+str(i)+'.h5')
#GENERAR MAPA FINAL DE CLASIFICACIÓN
print('dataOA = '+ str(vectOA)) 
print('OA = '+ str(OA/numTest)) 
datosSalida = classifier.predict(datosPrueba)
datosSalida = preparar.predictionToImage(datosSalida)
######################################  GRAFICAS  #####################################################################
#GROUND TRUTH vs OUTPUT
data.graficarHsi_VS(groundTruth, datosSalida)
logger.close()
#GRAFICAR TRAINING AND VALIDATION LOSS
data.graficar_history(history)