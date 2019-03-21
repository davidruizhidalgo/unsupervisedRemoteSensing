#PRUEBA DE RED CONVOLUCIONAL 2D - CLASIFICACION HSI 
#Se utiliza PCA para reduccion dimensional y estraccion de caracteristicas espectrales. A la red convolucional se introduce
#una ventana sxs de la imagen original para la generacion de caracteristicas espaciales a partir de la convolucion. 
#Se utiliza como capa de salida un clasificador tipo Multinomial logistic regression. Todas las capas utilizan entrenamiento supervisado. 
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
from sklearn.metrics import confusion_matrix, precision_score, recall_score, cohen_kappa_score

ventana = 9 #VENTANA 2D de PROCESAMIENTO
#CARGAR IMAGEN HSI Y GROUND TRUTH
data = CargarHsi('Indian_pines')
imagen = data.imagen
groundTruth = data.groundTruth

#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis()
imagenPCA = pca.pca_calculate(imagen, varianza=0.95) 

#ESTIMACIÓN DE EXTENDED ATTRIBUTE PROFILES
mp = morphologicalProfiles()
imagenEAP = mp.EAP(imagenPCA)

#PREPARAR DATOS PARA EJECUCIÓN
preparar = PrepararDatos(imagenEAP, groundTruth, False)
#CARGAR RED INCEPTION
model = load_model('hsiClassificationCNN2D.h5')
print(model.summary())

#GENERACION OA - Overall Accuracy 
datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)   #TOTAL MUESTRAS
test_loss, OA = model.evaluate(datosPrueba, etiquetasPrueba)            #EVALUAR MODELO

#GENERACION AA - Average Accuracy 
AA = 0 
ClassAA = np.zeros(groundTruth.max()+1)
for i in range(1,groundTruth.max()+1):                      #QUITAR 1 para incluir datos del fondo
    datosClase, etiquetasClase = preparar.extraerDatosClase2D(ventana,i) #MUESTRAS DE UNA CLASE
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

#GRAFICAS
data.graficarHsi_VS(groundTruth, datosSalida)
