#PRUEBA DE REDES PROFUNDAS SEMISUPERVISADAS.
# Se crea el archivo xxx_TEST.txt y se cargan las redes entrenadas para econtrar: 
    #Overall Accuracy 
    #Average Accuracy 
    #Kappa Coefficient
import warnings
warnings.filterwarnings('ignore')
from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from package.MorphologicalProfiles import morphologicalProfiles
from package.dataLogger import DataLogger

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn import svm
import numpy as np
import os 

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Add
from keras.layers import UpSampling2D, Dropout, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.models import Model
from keras.optimizers import SGD
from keras import regularizers

from keras import backend as K 
from keras import initializers
from keras.utils import plot_model
from keras.models import load_model
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score, cohen_kappa_score

from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def riemann_classifier(features_tr, etiquetasEntrenamiento, method='tan'):
    #Reshape features (n_samples, m_filters, p_features)
    features_tr = features_tr.reshape(features_tr.shape[0],features_tr.shape[1]*features_tr.shape[2],features_tr.shape[3])
    features_tr = np.transpose(features_tr, (0, 2, 1))
    #Reshape labels from categorical 
    etiquetasEntrenamiento = np.argmax(etiquetasEntrenamiento, axis=1)

    #Riemannian Classifier using Minimum Distance to Riemannian Mean
    covest = Covariances(estimator='lwf')
    if method == 'mdm':
        mdm = MDM()
        classifier = make_pipeline(covest,mdm)
    if method == 'tan':
        ts = TangentSpace()
        lda = LinearDiscriminantAnalysis()
        classifier = make_pipeline(covest,ts,lda)
    
    classifier.fit(features_tr, etiquetasEntrenamiento)
    return classifier

def accuracy(y_true, y_pred):
    if y_true.ndim>1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim>1:
        y_pred = np.argmax(y_pred, axis=1)

    OA = accuracy_score(y_true, y_pred)
    return OA

def reshapeFeatures(features_test):
    features_test = features_test.reshape(features_test.shape[0],features_test.shape[1]*features_test.shape[2],features_test.shape[3])
    features_test = np.transpose(features_test, (0, 2, 1))
    return features_test

#CARGAR IMAGEN HSI Y GROUND TRUTH
numTest = 10
riemann = False
dataSet = 'KSC'
test = 'pcaBCAE'  # pcaSCAE SCAE pcaBCAE BCAE
fe_eep = False     # false for PCA, true for EEP 

ventana = 8 #VENTANA 2D de PROCESAMIENTO
data = CargarHsi(dataSet)
imagen = data.imagen
groundTruth = data.groundTruth

#CREAR FICHERO DATA LOGGER 
logger = DataLogger(dataSet+'_TEST',test) 

#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis()
imagenFE = pca.pca_calculate(imagen, varianza=0.95)
#imagenFE = pca.pca_calculate(imagen, componentes=18)
print(imagenFE.shape)

#ESTIMACIÓN DE EXTENDED EXTINTION PROFILES
if fe_eep:    
    mp = morphologicalProfiles()
    imagenFE = mp.EEP(imagenFE, num_levels=4)    
    print(imagenFE.shape)

for i in range(0, numTest):
    #PREPARAR DATOS PARA EJECUCIÓN
    preparar = PrepararDatos(imagenFE, groundTruth, False)
    datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,20,ventana)
    datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)   #TOTAL MUESTRAS 

    #CARGAR MODELOS ENTRENADOS DE EXTRACIÓN DE CARACTERISTICAS
    encoder = load_model(os.path.join(logger.path,'FE_'+test+str(i)+'.h5'), custom_objects={'euclidean_distance_loss': euclidean_distance_loss}) 
    #Generar caracteristicicas con los datos de entrada
    features_tr = encoder.predict(datosEntrenamiento)
    features_test = encoder.predict(datosPrueba)
    
    #GENERACION OA - Overall Accuracy     
    if riemann: 
    ################################## CLASIFICADOR RIEMANN #########################################################
        classifier = riemann_classifier(features_tr, etiquetasEntrenamiento, method='tan')
        features_test = reshapeFeatures(features_test)
    #################################################################################################################
    else:
    ##################################CLASIFICADOR LOGISTIC REGRESSION###############################################
        classifier = load_model(os.path.join(logger.path,'C_'+test+str(i)+'.h5')) 
    #################################################################################################################
    datosSalida = classifier.predict(features_test)
    OA = accuracy(etiquetasPrueba, datosSalida)                #EVALUAR MODELO
 
    #GENERACION AA - Average Accuracy 
    AA = 0 
    ClassAA = np.zeros(groundTruth.max()+1)
    for j in range(1,groundTruth.max()+1):                      #QUITAR 1 para incluir datos del fondo
        datosClase, etiquetasClase = preparar.extraerDatosClase2D(ventana,j)    #MUESTRAS DE UNA CLASE                 
        features_class = encoder.predict(datosClase)
        if riemann:
            features_class = reshapeFeatures(features_class)                   # SOLO PARA RIEMMANIAN CLASSIFIER
        claseSalida = classifier.predict(features_class)
        ClassAA[j] = accuracy(etiquetasClase, claseSalida)                     #EVALUAR MODELO PARA LA CLASE
        AA += ClassAA[j]
    AA /= groundTruth.max()                                                    #SUMAR 1 para incluir datos del fondo

    #GENERACION Kappa Coefficient
    if not riemann:
        datosSalida = datosSalida.argmax(axis=1)
    etiquetasPrueba = etiquetasPrueba.argmax(axis=1)
    kappa = cohen_kappa_score(etiquetasPrueba, datosSalida)
    
    print('OA = '+ str(OA))                          #Overall Accuracy 
    print('AA = '+ str(AA)+' ='+ str(ClassAA))       #Average Accuracy 
    print('kappa = '+ str(kappa))                    #Kappa Coefficient
    
    #GENERACION DATA LOGGER 
    logger.savedataPerformance(OA, ClassAA, kappa)
    print('TESTING NET DONE :'+str(i))

#GENERAR MAPA FINAL DE CLASIFICACIÓN
datosSalida = classifier.predict(features_test)
imagenSalida = preparar.predictionToImage(datosSalida)
data.graficarHsi_VS(groundTruth, imagenSalida)
K.clear_session()
logger.close()
