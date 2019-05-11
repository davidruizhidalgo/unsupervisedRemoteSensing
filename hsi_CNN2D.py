#ENTRENAMIENTO DE RED CONVOLUCIONAL 2D - CLASIFICACION HSI 
#Se utiliza PCA para reduccion dimensional y estraccion de caracteristicas espectrales. A la red convolucional se introduce
#una ventana sxs de la imagen original para la generacion de caracteristicas espaciales a partir de la convolucion. 
#Se utiliza como capa de salida un clasificador tipo Multinomial logistic regression. Todas las capas utilizan entrenamiento supervisado. 
from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from package.MorphologicalProfiles import morphologicalProfiles
from keras import layers
from keras import models
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
from io import open

#CARGAR IMAGEN HSI Y GROUND TRUTH
numTest = 10
dataSet = 'PaviaU'
ventana = 9 #VENTANA 2D de PROCESAMIENTO
data = CargarHsi(dataSet)
imagen = data.imagen
groundTruth = data.groundTruth

nlogg = 'logger_'+dataSet+'.txt'
fichero = open(nlogg,'w')  
fichero.write('Datos EAP + CNN2D')

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
    datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,30,ventana)
    datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)

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
    history = model.fit(datosEntrenamiento,etiquetasEntrenamiento,epochs=25,batch_size=512,validation_data=(datosValidacion, etiquetasValidacion))

    #EVALUAR MODELO
    test_loss, test_acc = model.evaluate(datosPrueba, etiquetasPrueba)
    vectOA[i] = test_acc
    OA = OA+test_acc
    #LOGGER DATOS DE ENTRENAMIENTO
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    #CREAR DATA LOGGER
    fichero.write('\n'+str(loss))
    fichero.write('\n'+str(val_loss))
    fichero.write('\n'+str(acc))
    fichero.write('\n'+str(val_acc))
    #GUARDAR MODELO DE RED CONVOLUCIONAL
    model.save('hsiCNN2D'+str(i)+'.h5')

#GENERAR MAPA FINAL DE CLASIFICACIÓN
print('dataOA = '+ str(vectOA)) 
print('OA = '+ str(OA/numTest)) 
datosSalida = model.predict(datosPrueba)
datosSalida = preparar.predictionToImage(datosSalida)
#GRAFICAS
data.graficarHsi_VS(groundTruth, datosSalida)
fichero.close()

#######################################################
##GRAFICAR TRAINING AND VALIDATION LOSS
#plt.figure(1)
#plt.subplot(211)
#epochs = range(1, len(loss) + 1)
#plt.plot(epochs, loss, 'bo', label='Training loss')
#plt.plot(epochs, val_loss, 'b', label='Validation loss')
#plt.title('Training and validation loss')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()

#plt.subplot(212)
#plt.plot(epochs, acc, 'bo', label='Training acc')
#plt.plot(epochs, val_acc, 'b', label='Validation acc')
#plt.title('Training and validation accuracy')
#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.legend()
#plt.show()
##
