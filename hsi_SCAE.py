#CONVOLUTIONAL STACKED AUTOENCODER 
#Se implementa el encoder y el decoder convolucional para aprender una representación de las caracteristicas de los datos de entrada.
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
from keras.layers.core import Flatten
from keras.models import Model
from keras.optimizers import SGD
from keras import regularizers

#CARGAR IMAGEN HSI Y GROUND TRUTH
numTest = 1
dataSet = 'IndianPines'
ventana = 16 #VENTANA 2D de PROCESAMIENTO
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
mp = morphologicalProfiles()
imagenEAP = mp.EAP(imagenPCA, num_thresholds=6)  #####################
print(imagenEAP.shape)
OA = 0
vectOA = np.zeros(numTest)
for i in range(0, numTest):
    #PREPARAR DATOS PARA ENTRENAMIENTO
    preparar = PrepararDatos(imagenEAP, groundTruth, False)
    datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,20,ventana)
    datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)

    #STACKED CONVOLUTIONAL AUTOENCODER
    input_img = Input(shape=(datosEntrenamiento.shape[1],datosEntrenamiento.shape[2],datosEntrenamiento.shape[3]))  
    x = Conv2D(128, (5, 5), activation='relu', padding='same')(input_img)
    x = Conv2D(128, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
    encoded = Conv2D(32, (3, 3), strides=2, activation='relu', padding='same')(x)
    
    x = Conv2D(32, (5, 5), activation='relu', padding='same')(encoded)
    x = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(64, (5, 5), activation='relu', padding='same')(x)
    x = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    x = Conv2D(128, (5, 5), activation='relu', padding='same')(x)
    x = Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)
    decoded = Conv2D(datosEntrenamiento.shape[3], (5, 5), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    #print(autoencoder.summary())

    #TRAIN AUTOENCODER
    epochs = 30
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr = lrate, decay = decay, momentum = .7, nesterov = False)
    autoencoder.compile(optimizer=sgd, loss = 'binary_crossentropy', metrics=['accuracy']) #   loss=euclidean_distance_loss
    autoencoder.fit(datosEntrenamiento, datosEntrenamiento, epochs=epochs, batch_size=128, shuffle=True, validation_data=(datosValidacion, datosValidacion))

    #FINE TUNNING FULL CONECTED
    fullconected = Flatten()(encoded)
    fullconected = Dropout(.3)(fullconected)
    fullconected = Dense(groundTruth.max()+1, activation = 'softmax')(fullconected) 
   
    classifier = Model(inputs = input_img, outputs = fullconected) #GENERA EL MODELO FINAL
    #Se asignan los pesos del entrenamiento al modelo completo del Classifier
    classifier.layers[1].set_weights(autoencoder.layers[1].get_weights()) 
    classifier.layers[2].set_weights(autoencoder.layers[2].get_weights()) 
    classifier.layers[3].set_weights(autoencoder.layers[3].get_weights()) 
    classifier.layers[4].set_weights(autoencoder.layers[4].get_weights()) 
    classifier.layers[5].set_weights(autoencoder.layers[5].get_weights()) 
    classifier.layers[6].set_weights(autoencoder.layers[6].get_weights()) 
    #Congelar entrenamiento del encoder
    for layer in classifier.layers[1:-3]:
        #pass
        layer.trainable = False
    #print(classifier.summary())
    #ENTRENAMIENTO DE LA RED GENERAL
    classifier.compile(loss='categorical_crossentropy', optimizer = 'rmsprop', metrics=['accuracy'])
    history = classifier.fit(datosEntrenamiento, etiquetasEntrenamiento, epochs=epochs, batch_size=128, shuffle=True, validation_data=(datosValidacion, etiquetasValidacion))    #EVALUAR MODELO
    #VALIDACIÓN DE LA RED ENTRENADA
    test_loss, test_acc = classifier.evaluate(datosPrueba, etiquetasPrueba)
    vectOA[i] = test_acc
    OA = OA+test_acc
    #LOGGER DATOS DE ENTRENAMIENTO
    logger.savedataTrain(history)
    #GUARDAR MODELO DE RED CONVOLUCIONAL
    classifier.save('SCAE'+str(i)+'.h5')

#GENERAR MAPA FINAL DE CLASIFICACIÓN
print('dataOA = '+ str(vectOA)) 
print('OA = '+ str(OA/numTest)) 
datosSalida = classifier.predict(datosPrueba)
datosSalida = preparar.predictionToImage(datosSalida)
#GRAFICAS
data.graficarHsi_VS(groundTruth, datosSalida)
logger.close()

#################################################################################################################
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
