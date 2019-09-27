#STACKED CONVOLUTIONAL AUTOENCODER con capas de reconstrucción tipo refinement_layer
from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos

import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Conv2DTranspose, Add
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.models import Model
from keras.optimizers import SGD
from keras import regularizers

from keras import backend as K 
from keras import initializers
from keras.utils import plot_model

###############################FUNCIONES########################################################################################################################
def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def refinement_layer(lateral_tensor, vertical_tensor, num_filters, l2_loss):
    conv1 = Conv2D(num_filters, (3, 3), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(lateral_tensor)
    bchn1 = BatchNormalization()(conv1)
    conv2 = Conv2D(num_filters, (3, 3), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(bchn1)
    conv3 = Conv2D(num_filters, (3, 3), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(vertical_tensor)
    added = Add()([conv2, conv3])
    up = UpSampling2D(size=(2, 2), interpolation='nearest')(added)
    refinement = BatchNormalization()(up)
    return refinement

def cae(N , input_tensor, input_layer,nb_bands, l2_loss, top=True):
    encoded1_bn = BatchNormalization()(input_layer)
    encoded1 = Conv2D(N, (3, 3), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(encoded1_bn)
    encoded1_dg = MaxPooling2D((3,3), strides=2, padding='same')(encoded1)
    
    encoded2_bn = BatchNormalization()(encoded1_dg)
    encoded2 = Conv2D(2*N, (3, 3), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(encoded2_bn)
    encoded2_dg = MaxPooling2D((3,3), strides=2, padding='same')(encoded2)
    
    encoded3_bn = BatchNormalization()(encoded2_dg)
    encoded3 = Conv2D(2*N, (3, 3), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(encoded3_bn)
    encoded3_dg = MaxPooling2D((3,3), strides=2, padding='same')(encoded3)
    
    encoded4_bn = BatchNormalization()(encoded3_dg)
    encoded4 = Conv2D(4*N, (3, 3), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(encoded4_bn)
    
    refinement3 = refinement_layer(encoded3_dg, encoded4, 2*N, l2_loss)
    refinement2 = refinement_layer(encoded2_dg, refinement3, 2*N, l2_loss)
    refinement1 = refinement_layer(encoded1_dg, refinement2, N, l2_loss)
    
    if top: 
        output_tensor = Conv2D(nb_bands, (1, 1), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(refinement1)
        autoencoder = Model(input_tensor, output_tensor)
    else:
        autoencoder = Model(input_tensor, refinement1)
    return autoencoder

###########################PROGRAMA PRINCIPAL################################################################################################################

#CARGAR IMAGEN HSI Y GROUND TRUTH
numTest = 1
dataSet = 'IndianPines'
ventana = 8 #VENTANA 2D de PROCESAMIENTO
data = CargarHsi(dataSet)
imagen = data.imagen
groundTruth = data.groundTruth

for i in range(0, numTest):
    #PREPARAR DATOS PARA ENTRENAMIENTO
    preparar = PrepararDatos(imagen, groundTruth, False)
    datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,20,ventana)
    datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)
    ######################STACKED CONVOLUTIONAL AUTOENCODER#################################################################################################
    epochs = 50 #número de iteraciones
    input_img = Input(shape=(datosEntrenamiento.shape[1],datosEntrenamiento.shape[2],datosEntrenamiento.shape[3])) #tensor de entrada
    nd_scae = [64, 32, 16] #dimension de cada uno de los autoencoders
    #convolutional autoencoders
    len_i = 0
    encoder = input_img
    for i in range(len(nd_scae)):
        autoencoder = cae(nd_scae[i] , input_img, encoder, datosEntrenamiento.shape[3], 0.01, top=True)
        #congelar capas ya entrenadas
        if i != 0:
           for layer in autoencoder.layers[1:len_i +1]:
               layer.trainable = False
        print(autoencoder.summary())
        autoencoder.compile(optimizer='rmsprop', loss=euclidean_distance_loss, metrics=['accuracy']) #   loss='binary_crossentropy'
        autoencoder.fit(datosEntrenamiento, datosEntrenamiento, epochs=epochs, batch_size=128, shuffle=True, validation_data=(datosValidacion, datosValidacion))
        #Quitar capa de salida del autoencoder i
        autoencoder.layers.pop()
        len_i = len(autoencoder.layers)
        encoder = autoencoder.layers[-1].output
    ##################################CAPA DE SALIDA###############################################
    fullconected = Flatten()(encoder)
    fullconected = Dense(128, activation = 'relu')(fullconected) 
    fullconected = Dense(groundTruth.max()+1, activation = 'softmax')(fullconected) 
    classifier = Model(inputs = input_img, outputs = fullconected) #GENERA EL MODELO FINAL
    #Congela las capas de los autoencoders
    for layer in classifier.layers[1:-3]:
        layer.trainable = False
    print(classifier.summary())
    classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])   
    history = classifier.fit(datosEntrenamiento, etiquetasEntrenamiento, epochs=epochs, batch_size=128, shuffle=True, validation_data=(datosValidacion, etiquetasValidacion))

###########################GRAFICAS Y SALIDAS###############################
datosSalida = classifier.predict(datosPrueba)
datosSalida = preparar.predictionToImage(datosSalida)
data.graficarHsi_VS(groundTruth, datosSalida)
data.graficar_history(history)
plot_model(classifier, to_file='SCAEmodel.png')
K.clear_session()