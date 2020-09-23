#BRANCHES OF CONVOLUTIONAL STACKED AUTOENCODERS V2 - INCEPTION BASED  
#Se implementa el encoder y el decoder convolucional utilizando la arquitectura de la red INCEPTION y una capa
# de recosntrucción tipo refinement_layer para aprender una representación de las caracteristicas de los datos de entrada.
import warnings
warnings.filterwarnings('ignore')
from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from package.MorphologicalProfiles import morphologicalProfiles
from package.dataLogger import DataLogger

import matplotlib.pyplot as plt
import numpy as np
import os 

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Add, concatenate
from keras.layers import UpSampling2D, Dropout, Conv2DTranspose
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

def refinement_layer(lateral_tensor, vertical_tensor, kernel_size, num_filters, l2_loss):
    conv1 = Conv2D(num_filters, (kernel_size, kernel_size), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(lateral_tensor)
    bchn1 = BatchNormalization()(conv1)
    conv2 = Conv2D(num_filters, (kernel_size, kernel_size), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(bchn1)
    conv3 = Conv2D(num_filters, (kernel_size, kernel_size), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(vertical_tensor)
    added = Add()([conv2, conv3])
    up = UpSampling2D(size=(2, 2), interpolation='nearest')(added)
    refinement = BatchNormalization()(up)
    return refinement

def cae(N , input_tensor, nb_bands, kernel_size, l2_loss):
    encoded1_bn = BatchNormalization()(input_tensor)
    encoded1 = Conv2D(N, (kernel_size, kernel_size), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(encoded1_bn)
    encoded1_dg = MaxPooling2D((2,2), strides=2, padding='same')(encoded1)
    
    encoded2_bn = BatchNormalization()(encoded1_dg)
    encoded2 = Conv2D(2*N, (kernel_size, kernel_size), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(encoded2_bn)
    encoded2_dg = MaxPooling2D((2,2), strides=2, padding='same')(encoded2)
    
    encoded3_bn = BatchNormalization()(encoded2_dg)
    encoded3 = Conv2D(2*N, (kernel_size, kernel_size), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(encoded3_bn)
    encoded3_dg = MaxPooling2D((2,2), strides=2, padding='same')(encoded3)
    
    encoded4_bn = BatchNormalization()(encoded3_dg)
    encoded4 = Conv2D(4*N, (kernel_size, kernel_size), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(encoded4_bn)
    
    refinement3 = refinement_layer(encoded3_dg, encoded4, kernel_size, 2*N, l2_loss)
    refinement2 = refinement_layer(encoded2_dg, refinement3, kernel_size, 2*N, l2_loss)
    refinement1 = refinement_layer(encoded1_dg, refinement2, kernel_size, N, l2_loss)
    
    output_tensor = Conv2D(nb_bands, (1, 1), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(refinement1)
    autoencoder = Model(input_tensor, output_tensor)
    return autoencoder
###########################PROGRAMA PRINCIPAL###################################################################################################################
#CARGAR IMAGEN HSI Y GROUND TRUTH
numTest = 10
dataSet = 'KSC'
test =  'BCAE'     # pcaBCAE BCAE
fe_eep = True      # false for PCA, true for EEP 

ventana = 8 #VENTANA 2D de PROCESAMIENTO
data = CargarHsi(dataSet)
imagen = data.imagen
groundTruth = data.groundTruth

#CREAR FICHERO DATA LOGGER 
logger = DataLogger(dataSet, test) 

#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis()
imagenFE = pca.pca_calculate(imagen, varianza=0.95)
#imagenFE = pca.pca_calculate(imagen, componentes=4)
print(imagenFE.shape)

#ESTIMACIÓN DE EXTENDED EXTINTION PROFILES
if fe_eep:    
    mp = morphologicalProfiles()
    imagenFE = mp.EEP(imagenFE, num_levels=4)    
    print(imagenFE.shape)

OA = 0
vectOA = np.zeros(numTest)
for i in range(0, numTest):
    #PREPARAR DATOS PARA ENTRENAMIENTO
    preparar = PrepararDatos(imagenFE, groundTruth, False)
    datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,20,ventana)
    datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)
    ################################## CONVOLUTIONAL AUTOENCODERs################################################################################################
    epochs = 50 #número de iteraciones
    input_img = Input(shape=(datosEntrenamiento.shape[1],datosEntrenamiento.shape[2],datosEntrenamiento.shape[3])) #tensor de entrada
    nd_bcae = [1, 3, 5, 7] #kernel de cada uno de los autoencoders
    #convolutional autoencoders
    branches = []
    for j in range(len(nd_bcae)): 
        autoencoder = cae(16, input_img, datosEntrenamiento.shape[3], nd_bcae[j], 0.01)
        print(autoencoder.summary())
        autoencoder.compile(optimizer='rmsprop', loss=euclidean_distance_loss, metrics=['accuracy']) #   loss='binary_crossentropy'
        autoencoder.fit(datosEntrenamiento, datosEntrenamiento, epochs=epochs, batch_size=128, shuffle=True, validation_data=(datosValidacion, datosValidacion))
        #Quitar capa de salida del autoencoder j
        autoencoder.layers.pop()
        branches.append( autoencoder.layers[-1].output ) 
    output = concatenate(branches, axis=-1)
    
    ################################### BRANCHES OF STACKED AUTOENCODERS ##########################################################################################
    encoder = Model(inputs = input_img, outputs = output)
    encoder.save(os.path.join(logger.path,'FE_'+test+str(i)+'.h5'))
    features = encoder.predict(datosEntrenamiento)
    features_val = encoder.predict(datosValidacion)
    ##################################CLASIFICADOR CAPA DE SALIDA###############################################
    input_features = Input(shape=(features.shape[1],features.shape[2],features.shape[3]))
    fullconected = Flatten()(input_features)
    fullconected = Dense(128, activation = 'relu')(fullconected) 
    fullconected = Dense(groundTruth.max()+1, activation = 'softmax')(fullconected) 
    classifier = Model(inputs = input_features, outputs = fullconected) #GENERA EL MODELO FINAL

    print(classifier.summary())
    classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])   
    history = classifier.fit(features, etiquetasEntrenamiento, epochs=epochs, batch_size=128, shuffle=True, validation_data=(features_val, etiquetasValidacion))
    #VALIDACIÓN DE LA RED ENTRENADA
    features_out = encoder.predict(datosPrueba)
    test_loss, test_acc = classifier.evaluate(features_out, etiquetasPrueba)
    vectOA[i] = test_acc
    OA = OA+test_acc
    #LOGGER DATOS DE ENTRENAMIENTO
    logger.savedataTrain(history)
    #GUARDAR MODELO DE RED CONVOLUCIONAL
    classifier.save(os.path.join(logger.path,'C_'+test+str(i)+'.h5'))
###########################GRAFICAS Y SALIDAS###############################
plot_model(encoder, to_file=os.path.join(logger.path,test+'model.png'))
datosSalida = classifier.predict(features_out)
datosSalida = preparar.predictionToImage(datosSalida)
data.graficarHsi_VS(groundTruth, datosSalida)
data.graficar_history(history)
K.clear_session()
logger.close()
