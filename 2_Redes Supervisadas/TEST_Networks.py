#PRUEBA DE REDES PROFUNDAS SUPERVISADAS.
# Se crea el archivo xxx_TEST.txt y se carga las redes entrenadas para econtrar: 
    #Overall Accuracy 
    #Average Accuracy 
    #Kappa Coefficient
# Cambiar cargar datos 2D o 3D dependiendo de la prueba realizada

from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from package.MorphologicalProfiles import morphologicalProfiles
from package.dataLogger import DataLogger
from keras import layers 
from keras import models
from keras.models import load_model
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, cohen_kappa_score

#CARGAR IMAGEN HSI Y GROUND TRUTH
numTest = 10
dataSet = 'PaviaU'                                     #==========================> CAMBIAR DE ACUERDO A LA PRUEBA REALIZADA
ventana = 9 #VENTANA 2D de PROCESAMIENTO
data = CargarHsi(dataSet)
imagen = data.imagen
groundTruth = data.groundTruth

#CREAR FICHERO DATA LOGGER 
logger = DataLogger(dataSet+'_TEST')                          #==========================> CAMBIAR DE ACUERDO A LA PRUEBA REALIZADA

#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis()
#imagenPCA = pca.pca_calculate(imagen, varianza=0.95)        #==========================> CAMBIAR DE ACUERDO A LA PRUEBA REALIZADA
#imagenPCA = pca.pca_calculate(imagen, componentes=4)
imagenPCA = pca.kpca_calculate(imagen, componentes = 15)
print('K-PCA DONE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')


#ESTIMACIÓN DE EXTENDED ATTRIBUTE PROFILES
#mp = morphologicalProfiles()
#imagenEAP = mp.EAP(imagenPCA, num_thresholds=6)                               #==========================> CAMBIAR DE ACUERDO A LA PRUEBA REALIZADA

for i in range(0, numTest):
    #PREPARAR DATOS PARA EJECUCIÓN
    preparar = PrepararDatos(imagenPCA, groundTruth, False)
    #CARGAR RED INCEPTION
    model = load_model('kpcaINCEPTION'+str(i)+'.h5')     #==========================> CAMBIAR DE ACUERDO A LA PRUEBA REALIZADA 
    print(model.summary())                                  #  ('hsiCNN2D'+str(i)+'.h5')  ('kpcaCNN'+str(i)+'.h5')  ('kpcaINCEPTION'+str(i)+'.h5') 

 #GENERACION OA - Overall Accuracy 
    datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)   #TOTAL MUESTRAS                         #==========================> CAMBIAR DE ACUERDO A LA PRUEBA REALIZADA
    test_loss, OA = model.evaluate(datosPrueba, etiquetasPrueba)            #EVALUAR MODELO

    #GENERACION AA - Average Accuracy 
    AA = 0 
    ClassAA = np.zeros(groundTruth.max()+1)
    for i in range(1,groundTruth.max()+1):                      #QUITAR 1 para incluir datos del fondo
        datosClase, etiquetasClase = preparar.extraerDatosClase2D(ventana,i) #MUESTRAS DE UNA CLASE                 #==========================> CAMBIAR DE ACUERDO A LA PRUEBA REALIZADA
        test_loss, ClassAA[i] = model.evaluate(datosClase, etiquetasClase)      #EVALUAR MODELO PARA LA CLASE
        AA += ClassAA[i]
    AA /= groundTruth.max()                                     #SUMAR 1 para incluir datos del fondo

    #GENERAR MAPA FINAL DE CLASIFICACIÓN
    datosSalida = model.predict(datosPrueba)
    etiquetasPred = datosSalida.copy()
    datosSalida = preparar.predictionToImage(datosSalida)      

    #GENERACION Kappa Coefficient
    etiquetasPred = etiquetasPred.argmax(axis=1)
    etiquetasPrueba = etiquetasPrueba.argmax(axis=1)
    kappa = cohen_kappa_score(etiquetasPrueba, etiquetasPred)

    print('OA = '+ str(OA))                          #Overall Accuracy 
    print('AA = '+ str(AA)+' ='+ str(ClassAA))       #Average Accuracy 
    print('kappa = '+ str(kappa))                    #Kappa Coefficient
    
    #GENERACION DATA LOGGER 
    logger.savedataPerformance(OA, ClassAA, kappa)
    
logger.close()