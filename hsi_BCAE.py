#BRANCHES OF CONVOLUTIONAL STACKED AUTOENCODERS - INCEPTION BASED  
#Se implementa el encoder y el decoder convolucional utilizando la arquitectura de la red INCEPTION
# para aprender una representación de las caracteristicas de los datos de entrada.
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

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Conv2DTranspose, concatenate
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
imagenEEP = mp.EEP(imagenPCA, num_levels=10)    
print(imagenEEP.shape)

OA = 0
vectOA = np.zeros(numTest)
for i in range(0, numTest):
    #PREPARAR DATOS PARA ENTRENAMIENTO
    preparar = PrepararDatos(imagenEEP, groundTruth, False)
    datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,30,ventana)
    datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)
    #########################################################################################################################
    ######################## BRANCH CONVOLUTIONAL AUTOENCODERS ################################################################
    #DEFINICION Y ENTRENAMIENTO CAPA POR CAPA DEL INCEPTION AUTOENCODER 
    input_img = Input(shape=(datosEntrenamiento.shape[1], datosEntrenamiento.shape[2], datosEntrenamiento.shape[3])) #Formato tensor de entrada
    #RAMA 1
    encoded_br1 = Conv2D(16, (1,1), strides=2, activation='relu', kernel_regularizer=regularizers.l2(0.001))(input_img)
    br = Conv2DTranspose(16, (1,1), strides=2, activation='relu', kernel_regularizer=regularizers.l2(0.001))(encoded_br1)
    decoded_br1 = Conv2D(datosEntrenamiento.shape[3], (3, 3), activation='sigmoid', padding='same', kernel_regularizer=regularizers.l2(0.001))(br)
    autoencoder1 = Model(input_img, decoded_br1)
    print(autoencoder1.summary())
    #RAMA 2
    br = Conv2D(16, (1,1), strides=1,activation='relu', kernel_regularizer=regularizers.l2(0.001))(input_img)
    encoded_br2 = Conv2D(8, (3,3), strides=2, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(br)
    br = Conv2DTranspose(8, (3,3), strides=2, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(encoded_br2)
    br = Conv2DTranspose(16, (1,1), strides=1,activation='relu', kernel_regularizer=regularizers.l2(0.001))(br)
    decoded_br2 = Conv2D(datosEntrenamiento.shape[3], (3, 3), activation='sigmoid', padding='same', kernel_regularizer=regularizers.l2(0.001))(br)
    autoencoder2 = Model(input_img, decoded_br2)
    print(autoencoder2.summary())
    #RAMA 3
    br = Conv2D(16,(3,3), strides=2, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(input_img)
    encoded_br3 = Conv2D(16, (3,3), strides=1, activation='relu', padding='same',  kernel_regularizer=regularizers.l2(0.001))(br)
    br = Conv2D(16, (3,3), strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(encoded_br3)
    br = Conv2DTranspose(16,(3,3), strides=2, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(br)
    decoded_br3 = Conv2D(datosEntrenamiento.shape[3], (3, 3), activation='sigmoid', padding='same', kernel_regularizer=regularizers.l2(0.001))(br)
    autoencoder3 = Model(input_img, decoded_br3)
    print(autoencoder3.summary())
    #RAMA 4
    br = Conv2D(32, (1, 1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(input_img)
    br = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(br)
    encoded_br4 = Conv2D(8, (3,3), strides=2, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(br)
    br = Conv2DTranspose(8, (3,3), strides=2, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(encoded_br4)
    br = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(br)
    br = Conv2D(32, (1, 1), activation='relu', kernel_regularizer=regularizers.l2(0.001))(br)
    decoded_br4 = Conv2D(datosEntrenamiento.shape[3], (3, 3), activation='sigmoid', padding='same', kernel_regularizer=regularizers.l2(0.001))(br)
    autoencoder4 = Model(input_img, decoded_br4)
    print(autoencoder4.summary())
    #PARAMETROS DE ENTRENAMIENTO
    epochs = 50
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr = lrate, decay = decay, momentum = .75, nesterov = True)
    autoencoder1.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    autoencoder2.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    autoencoder3.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    autoencoder4.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    #ENTRENAMIENTO DE CADA RAMA
    autoencoder1.fit(datosEntrenamiento, datosEntrenamiento, epochs = epochs, batch_size = 128, validation_data=(datosValidacion, datosValidacion), shuffle = True)
    autoencoder2.fit(datosEntrenamiento, datosEntrenamiento, epochs = epochs, batch_size = 128, validation_data=(datosValidacion, datosValidacion), shuffle = True)
    autoencoder3.fit(datosEntrenamiento, datosEntrenamiento, epochs = epochs, batch_size = 128, validation_data=(datosValidacion, datosValidacion), shuffle = True)
    autoencoder4.fit(datosEntrenamiento, datosEntrenamiento, epochs = epochs, batch_size = 128, validation_data=(datosValidacion, datosValidacion), shuffle = True)
    ########################################## INCEPTION AUTOENCODER AND FINE TUNNING ##############################################################################
    #ESTRUCTURA GENERAL INCEPTION AUTOENCODER
    # Rama A
    branch_a = Conv2D(16, (1,1), strides=2, activation='relu', kernel_regularizer=regularizers.l2(0.001))(input_img)
    # Rama B
    branch_b = Conv2D(16, (1,1), strides=1,activation='relu', kernel_regularizer=regularizers.l2(0.001))(input_img)
    branch_b = Conv2D(8, (3,3), strides=2, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(branch_b)
    # Rama C
    branch_c = Conv2D(16,(3,3), strides=2, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(input_img)
    branch_c = Conv2D(16, (3,3), strides=1, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(branch_c)
    # Rama D
    branch_d = Conv2D(32, (1,1), activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(input_img)
    branch_d = Conv2D(16, (3,3), activation='relu', padding='same',kernel_regularizer=regularizers.l2(0.001))(branch_d)
    branch_d = Conv2D(8, (3,3), strides=2, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(branch_d)
    # Se concatenan todas las rama  para tener un solo modelo en output
    output = concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)
    # Se añade como capa final de salida un clasificador tipo Multinomial logistic regression
    output = Flatten()(output)
    out    = Dense(groundTruth.max()+1, activation='softmax', kernel_regularizer=regularizers.l2(0.001))(output)
    # Se define el modelo total de la red 
    inception = Model(inputs = input_img, outputs = out)
    #COPIAR y CONGELAR LOS PESOS DE LAS CAPAS ENTRENADAS
    inception.layers[5].set_weights(autoencoder1.layers[1].get_weights()) 
    inception.layers[2].set_weights(autoencoder2.layers[1].get_weights()) 
    inception.layers[6].set_weights(autoencoder2.layers[2].get_weights()) 
    inception.layers[3].set_weights(autoencoder3.layers[1].get_weights()) 
    inception.layers[7].set_weights(autoencoder3.layers[2].get_weights()) 
    inception.layers[1].set_weights(autoencoder4.layers[1].get_weights()) 
    inception.layers[4].set_weights(autoencoder4.layers[2].get_weights()) 
    inception.layers[8].set_weights(autoencoder4.layers[3].get_weights()) 
    #Congelar entrenamiento del encoder
    for layer in inception.layers[1:-3]:
        layer.trainable = False
    #Imprimir modelo completo
    print(inception.summary())
    #################################### ENTRENAMIENTO MODELO COMPLETO #################################################
    inception.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    history = inception.fit(datosEntrenamiento, etiquetasEntrenamiento, epochs = epochs, batch_size = 128, validation_data=(datosValidacion, etiquetasValidacion), shuffle = True)
    ################################## VALIDACIÓN DE LA RED ENTRENADA ##################################################
    test_loss, test_acc = inception.evaluate(datosPrueba, etiquetasPrueba)
    vectOA[i] = test_acc
    OA = OA+test_acc
    #LOGGER DATOS DE ENTRENAMIENTO
    logger.savedataTrain(history)
    #GUARDAR MODELO DE RED CONVOLUCIONAL
    inception.save('BCAE'+str(i)+'.h5')
#GENERAR MAPA FINAL DE CLASIFICACIÓN
print('dataOA = '+ str(vectOA)) 
print('OA = '+ str(OA/numTest)) 
datosSalida = inception.predict(datosPrueba)
datosSalida = preparar.predictionToImage(datosSalida)
######################################  GRAFICAS  #####################################################################
#GROUND TRUTH vs OUTPUT
data.graficarHsi_VS(groundTruth, datosSalida)
logger.close()
#GRAFICAR TRAINING AND VALIDATION LOSS
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure(1)
plt.subplot(211)
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.subplot(212)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
