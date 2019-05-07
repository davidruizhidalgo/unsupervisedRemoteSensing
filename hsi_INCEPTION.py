#RED PROFUNDA CON TOPOLOGIA DE GRAFO INCEPTION - CLASIFICACION HSI 
#Se utiliza la topologia de red de grafos INCEPTION para la inclusion de caracteristicas espectrales y espaciales en la arquitectura de 
#la red profunda. Esto se logra utilizando redes convolucionales con diferentes tamaños de ventana; 1x1 para el manejo de caracteristicas 
#espectrales y 3x3 o 2x2 para el manejo de posibles dependencias espaciales. Se extrae entonces un tensor 4D utilizando una ventana sxs 
#de la imagen original.
#Se utiliza como capa de salida un clasificador tipo Multinomial logistic regression. Todas las capas utilizan entrenamiento supervisado. 
#Para el entrenamiento se utiliza el algoritmo de optimización de gradiente descendente estocastico con parametros variables. 
from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from package.MorphologicalProfiles import morphologicalProfiles 
from keras import layers
from keras.models import Model
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
from io import open

#CARGAR IMAGEN HSI Y GROUND TRUTH
numTest = 10
dataSet = 'IndianPines'
ventana = 9 #VENTANA 2D de PROCESAMIENTO
data = CargarHsi(dataSet)
imagen = data.imagen
groundTruth = data.groundTruth

#nlogg = 'logger_'+dataSet+'.txt'
#fichero = open(nlogg,'w')  
#fichero.write('Datos EAP + INCEPTION')

#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis()
imagenPCA = pca.pca_calculate(imagen, varianza=0.95)
#imagenPCA = pca.pca_calculate(imagen, componentes=4)
print(imagenPCA.shape)

#ESTIMACIÓN DE EXTENDED ATTRIBUTE PROFILES
mp = morphologicalProfiles()
imagenEAP = mp.EAP(imagenPCA, num_thresholds=10)
print(imagenEAP.shape)
OA = 0
vectOA = np.zeros(numTest)
for i in range(0, numTest):
    #PREPARAR DATOS PARA ENTRENAMIENTO
    preparar = PrepararDatos(imagenEAP, groundTruth, False)
    datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,30,ventana)
    datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)

    #DEFINICION RED CONVOLUCIONAL INCEPTION 
    input_tensor = Input(shape=(datosEntrenamiento.shape[1],datosEntrenamiento.shape[2],datosEntrenamiento.shape[3]))
    # Cada rama tiene el mismo estado de padding='same', lo cual es necesario para mantener todas las salidas de las ramas 
    # en el mismo tamaño. Esto posibilita la ejecución de la instrucción concatenate.
    # Rama A
    branch_a = layers.Conv2D(64, (1,1), activation='relu', padding='same')(input_tensor)
    # Rama B
    branch_b = layers.Conv2D(64, (1,1), activation='relu', padding='same')(input_tensor)
    branch_b = layers.Conv2D(64, (3,3), activation='relu', padding='same')(branch_b)
    # Rama C
    branch_c = layers.AveragePooling2D((3,3), strides=(1,1), padding='same')(input_tensor)
    branch_c = layers.Conv2D(64, (3,3), activation='relu', padding='same')(branch_c)
    # Rama D
    branch_d = layers.Conv2D(64, (1,1), activation='relu', padding='same')(input_tensor)
    branch_d = layers.Conv2D(64, (3,3), activation='relu', padding='same')(branch_d)
    branch_d = layers.Conv2D(64, (3,3), activation='relu', padding='same')(branch_d)
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
    print(test_acc)
    vectOA[i] = test_acc
    OA = OA+test_acc
    #LOGGER DATOS DE ENTRENAMIENTO
    #loss = history.history['loss']
    #val_loss = history.history['val_loss']
    #acc = history.history['acc']
    #val_acc = history.history['val_acc']
    #CREAR DATA LOGGER
    #fichero.write('\n'+str(loss))
    #fichero.write('\n'+str(val_loss))
    #fichero.write('\n'+str(acc))
    #fichero.write('\n'+str(val_acc))
    #GUARDAR MODELO DE RED CONVOLUCIONAL
    #model.save('hsiINCEPTION'+str(i)+'.h5')

#GENERAR MAPA FINAL DE CLASIFICACIÓN
print('dataOA = '+ str(vectOA)) 
print('OA = '+ str(OA/numTest)) 
datosSalida = model.predict(datosPrueba)
datosSalida = preparar.predictionToImage(datosSalida)
#GRAFICAS
data.graficarHsi_VS(groundTruth, datosSalida)
#fichero.close()
