# Numero de PCA y red SCAE V2
from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from sklearn.metrics import cohen_kappa_score
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Add
from keras.layers import UpSampling2D, Dropout, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.models import Model
from keras.optimizers import SGD
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from keras import backend as K 
from keras import initializers

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

def cae(N , input_tensor, input_layer,nb_bands, l2_loss):
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
    
    output_tensor = Conv2D(nb_bands, (1, 1), activation='relu', strides=1, padding='same', kernel_regularizer=regularizers.l2(l2_loss), kernel_initializer=initializers.glorot_normal())(refinement1)
    autoencoder = Model(input_tensor, output_tensor)
    return autoencoder
 ########################################################################################################################################################   
###########################PROGRAMA PRINCIPAL################################################################################################################
#CARGAR IMAGEN HSI Y GROUND TRUTH
numTest = 10
dataSet = 'Urban'
ventana = 8 #VENTANA 2D de PROCESAMIENTO
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
######################STACKED CONVOLUTIONAL AUTOENCODER#################################################################################################
    epochs = 35  #número de iteraciones
    input_img = Input(shape=(datosEntrenamiento.shape[1],datosEntrenamiento.shape[2],datosEntrenamiento.shape[3])) #tensor de entrada
    nd_scae = [64, 32, 16] #dimension de cada uno de los autoencoders
    #convolutional autoencoders
    len_i = 0
    encoder = input_img
    for j in range(len(nd_scae)):
        autoencoder = cae(nd_scae[j] , input_img, encoder, datosEntrenamiento.shape[3], 0.01)
        #congelar capas ya entrenadas
        if j != 0:
           for layer in autoencoder.layers[1:len_i +1]:
               layer.trainable = False
        print(autoencoder.summary())
        autoencoder.compile(optimizer='rmsprop', loss=euclidean_distance_loss, metrics=['accuracy']) #   loss='binary_crossentropy'
        autoencoder.fit(datosEntrenamiento, datosEntrenamiento, epochs=epochs, batch_size=128, shuffle=True, validation_data=(datosValidacion, datosValidacion))
        #Quitar capa de salida del autoencoder j
        autoencoder.layers.pop()
        len_i = len(autoencoder.layers)
        encoder = autoencoder.layers[-1].output
    #######################MODELO STACKED AUTOENCODER##################################################################
    encoder = Model(inputs = autoencoder.input, outputs = autoencoder.layers[-1].output)
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
    #EVALUAR MODELO DE RED CONVOLUCIONAL
    #GENERACION OA - Overall Accuracy 
    datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)   #TOTAL MUESTRAS   
    features_out = encoder.predict(datosPrueba)                   
    test_loss, OA = classifier.evaluate(features_out, etiquetasPrueba)       #EVALUAR MODELO
    vectOA[i] = OA                                                          #Almacena el OA de cada prueba
    #GENERACION AA - Average Accuracy 
    AA = 0 
    ClassAA = np.zeros(groundTruth.max()+1)
    for j in range(1,groundTruth.max()+1):                                        #QUITAR 1 para incluir datos del fondo
        datosClase, etiquetasClase = preparar.extraerDatosClase2D(ventana,j)      #MUESTRAS DE UNA CLASE                 
        features_class = encoder.predict(datosClase) 
        test_loss, ClassAA[j] = classifier.evaluate(features_class, etiquetasClase)   #EVALUAR MODELO PARA LA CLASE
        AA += ClassAA[j]
    AA /= groundTruth.max()                                                  #SUMAR 1 para incluir datos del fondo
    vectAA[i] = AA                                                           #Almacena el AA de cada prueba
    
    #GENERACION Kappa Coefficient
    datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)   #TOTAL MUESTRAS   
    features_out = encoder.predict(datosPrueba)     
    datosSalida = classifier.predict(features_out)
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