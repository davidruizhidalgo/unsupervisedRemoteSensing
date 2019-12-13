# Numero de K-PCA y red Inception
from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from sklearn.metrics import cohen_kappa_score
from keras import layers
from keras.models import Model
from keras.layers import Input
from keras.layers import Flatten
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from keras import backend as K 

#CARGAR IMAGEN HSI Y GROUND TRUTH
numTest = 10
dataSet = 'Pavia'
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
    imagenPCA = pca.kpca_calculate(imagen, componentes= 2*i+2) 
    print(imagenPCA.shape)
    print('DIMENSIONAL REDUCTION DONE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    preparar = PrepararDatos(imagenPCA, groundTruth, False)
    datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,30,ventana)

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
    epochs = 35
    lrate = 0.01
    decay = lrate/epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    history = model.fit(datosEntrenamiento,etiquetasEntrenamiento,epochs=epochs,batch_size=256,validation_data=(datosValidacion, etiquetasValidacion))

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