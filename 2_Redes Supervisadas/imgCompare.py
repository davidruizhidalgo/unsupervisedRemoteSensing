#COMPARACION DE IMAGENES
# El archivo carga una red neuronal y compara por medio de 
# una resta el groundthuth con el mapa de clasificación resultante.

from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from package.MorphologicalProfiles import morphologicalProfiles
from keras import layers 
from keras import models
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical

#CARGAR IMAGEN HSI Y GROUND TRUTH
dataSet = 'IndianPines'                                     #==========================> CAMBIAR DE ACUERDO A LA PRUEBA REALIZADA
ventana = 9 #VENTANA 2D de PROCESAMIENTO
data = CargarHsi(dataSet)
imagen = data.imagen
groundTruth = data.groundTruth

#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis()
imagenPCA = pca.pca_calculate(imagen, varianza=0.95)        #==========================> CAMBIAR DE ACUERDO A LA PRUEBA REALIZADA
#imagenPCA = pca.pca_calculate(imagen, componentes=4)

#ESTIMACIÓN DE EXTENDED ATTRIBUTE PROFILES
mp = morphologicalProfiles()
imagenEAP = mp.EAP(imagenPCA, num_thresholds=6)                               #==========================> CAMBIAR DE ACUERDO A LA PRUEBA REALIZADA

#PREPARAR DATOS PARA EJECUCIÓN
preparar = PrepararDatos(imagenEAP, groundTruth, False)
#CARGAR RED NEURONAL
model = load_model('hsiINCEPTION'+'0.h5')                                  #==========================> CAMBIAR DE ACUERDO A LA PRUEBA REALIZADA
print(model.summary())

 #GENERACION OA - Overall Accuracy 
datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)   #TOTAL MUESTRAS                         
test_loss, OA = model.evaluate(datosPrueba, etiquetasPrueba)            #EVALUAR MODELO
print('OA = '+ str(OA))                                                 #Overall Accuracy 

#GENERAR MAPA FINAL DE CLASIFICACIÓN
datosSalida = model.predict(datosPrueba)
etiquetasPred = datosSalida.copy()
datosSalida = preparar.predictionToImage(datosSalida)   

#COMPARACION
dataCompare = np.absolute(groundTruth-datosSalida)
dataCompare = (dataCompare > 0) * 1 

#GRAFICAS
data.graficarHsi_VS(datosSalida, dataCompare, cmap ='bw')