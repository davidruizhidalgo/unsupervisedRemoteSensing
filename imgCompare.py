#COMPARACION DE IMAGENES
# El archivo carga una red neuronal y compara por medio de 
# una resta el groundthuth con el mapa de clasificación resultante.
import warnings
warnings.filterwarnings('ignore')
from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from package.MorphologicalProfiles import morphologicalProfiles
from package.dataLogger import DataLogger
from keras import layers 
from keras import models
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical
from keras import backend as K 
import os 

def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

#CARGAR IMAGEN HSI Y GROUND TRUTH
dataSet = 'Pavia'                                                   #==========================> CAMBIAR DE ACUERDO A LA PRUEBA REALIZADA
test = 'pcaSCAE'         # carpeta logger: ex. pcaSCAE pcaCNN2D kpcaInception 
fe_eep = False           # false for PCA, true for EEP 
autoencoder = True       # false for pcaCNN2d and kpcaInception 

ventana = 8 #VENTANA 2D de PROCESAMIENTO
data = CargarHsi(dataSet)
imagen = data.imagen
groundTruth = data.groundTruth
#CREAR Path del DATA LOGGER 
logger = DataLogger(dataSet+'_TEST',test, False) 

#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis()                                         #==========================> CAMBIAR DE ACUERDO A LA PRUEBA REALIZADA
imagenFE = pca.pca_calculate(imagen, componentes=4)       #9, 4, 6, 4, 4, 18    
#imagenFE = pca.kpca_calculate(imagen, componentes=10)    #17, 4, 10, 4, 10, 15

#ESTIMACIÓN DE EXTENDED ATTRIBUTE PROFILES
if fe_eep:
    mp = morphologicalProfiles()
    imagenFE = mp.EAP(imagenFE, num_thresholds=6)                               #==========================> CAMBIAR DE ACUERDO A LA PRUEBA REALIZADA

#PREPARAR DATOS PARA EJECUCIÓN
preparar = PrepararDatos(imagenFE, groundTruth, False)
datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)           #TOTAL MUESTRAS   
#CARGAR RED NEURONAL
if autoencoder:
    encoder = load_model(os.path.join(logger.path,'FE_'+test+'9.h5'), custom_objects={'euclidean_distance_loss': euclidean_distance_loss}) 
    datosPrueba = encoder.predict(datosPrueba)
    model = load_model(os.path.join(logger.path,'C_'+test+'9.h5')) 
else:
    model = load_model(os.path.join(logger.path,test+'9.h5'))                       #==========================> CAMBIAR DE ACUERDO A LA PRUEBA REALIZADA                          

print(model.summary())
 #GENERACION OA - Overall Accuracy 
                      
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
K.clear_session()