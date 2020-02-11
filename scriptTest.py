#Script utilizado para el desarrollo de pruebas en el codigo. 
####### EN DESARROLLO ##########################
# 1. Desarrollar una arquitectura de red selección de caracteristicas NO SUPERVISADA - Capa de Salida FINE-TUNNING
        # CNN Autoendoder and full conected network +/- 75% acc
            #Desempeño con EEP:  +/- 85% acc
            #SVM lineal and rbf  +/- 92% acc  and +/- 85% acc
            #Riemannian Geometry +/- 80% acc
        # Grafo de CNN autoencoders and full conected network +/- 80% acc
            #Desempeño con EEP:  +/- 96% acc
            #SVM lineal and rbf  +/- 98% acc  and +/- 90% acc
            #Riemannian Geometry +/- 99% acc
        # Evaluar modificaciones en la funcion de costo
        # Evaluar estrategias de concatenacion en BSCAE

# 2. Evaluar un esquema de data augmentation => prepararDatos.py
# 4. Revisar documentacion reciente del estado del arte.

# pylint: disable=E1136  # pylint/issues/3139

from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from package.MorphologicalProfiles import morphologicalProfiles
from package.dataLogger import DataLogger
from keras.layers import Input, Dense, Conv2D, Flatten, Dropout, concatenate
from keras.layers import UpSampling2D, Dropout, Conv2DTranspose, MaxPooling2D, Add
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import regularizers
from keras import backend as K 
from keras import layers
from keras.optimizers import SGD
from keras import initializers
from sklearn.metrics import cohen_kappa_score
from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
import os 

#############################################################################################################################################################
###########################FUNCIONES UTILIZADAS##############################################################################################################
def reshapeFeaturesSVM(features_test):
    features_test = features_test.reshape(features_test.shape[0],features_test.shape[1]*features_test.shape[2],features_test.shape[3])
    features_test = features_test.reshape(features_test.shape[0],features_test.shape[1]*features_test.shape[2])
    return features_test

def reshapeFeaturesRIE(features_test):
    features_test = features_test.reshape(features_test.shape[0],features_test.shape[1]*features_test.shape[2],features_test.shape[3])
    features_test = np.transpose(features_test, (0, 2, 1))
    return features_test

def svm_classifier(features_tr, etiquetasEntrenamiento, kernel='linear'):
    #Reshape features (n_samples, m_features)
    features_tr = features_tr.reshape(features_tr.shape[0],features_tr.shape[1]*features_tr.shape[2],features_tr.shape[3])
    features_tr = features_tr.reshape(features_tr.shape[0],features_tr.shape[1]*features_tr.shape[2])
    #Reshape labels from categorical 
    etiquetasEntrenamiento = np.argmax(etiquetasEntrenamiento, axis=1)
    #SVM Classifier one-against-one
    classifier = svm.SVC(gamma='scale', decision_function_shape='ovo', kernel=kernel, verbose=False)
    classifier.fit(features_tr,etiquetasEntrenamiento)
    return classifier

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

#############################################################################################################################################################
###########################PROGRAMA PRINCIPAL################################################################################################################
###########################CARGAR IMAGEN HSI, GROUND TRUTH e INICIAR LOGGER##################################################################################
numTest = 10
dataSet = 'IndianPines'
test = 'eapInception'   # pcaInception eapInception
save = True             # false to avoid create logger
fe_eap = True           # false for PCA, true for EAP 

data = CargarHsi(dataSet)
imagen = data.imagen
groundTruth = data.groundTruth
print(imagen.shape)
###########################INICIAL FEATURE EXTRACTION########################################################################################################
#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis()
imagenFE = pca.pca_calculate(imagen, varianza=0.95)
#imagenFE = pca.pca_calculate(imagen, componentes=10)
print(imagenFE.shape)

#ESTIMACIÓN DE EXTENDED EXTINTION PROFILES
if fe_eap:    
    mp = morphologicalProfiles()
    imagenFE = mp.EAP(imagenFE, num_thresholds=6)    
    print(imagenFE.shape)


#############################################################################################################################################################
############################DEEP NEURAL NETWORK MODEL DEFINITION#############################################################################################
epochs = 35
########################CREAR FICHERO DATA LOGGER############################################################################################################ 
logger = DataLogger(fileName = dataSet+'_LRC', folder = test, save = True)      
########################PREPARAR DATOS PARA ENTRENAMIENTO####################################################################################################
ventana = 9  #VENTANA 2D de PROCESAMIENTO
preparar = PrepararDatos(imagenFE, groundTruth, False)
datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,30,ventana)
datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)

############################################DEFINICION RED CONVOLUCIONAL INCEPTION########################################################################### 
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
#############################################################################################################################################################
################################## CLASIFICADOR LOGISTIC REGRESION ##########################################################################################
# Se añade como capa final de salida un clasificador tipo Multinomial logistic regression
output = Flatten()(output)
output = Dense(groundTruth.max()+1, activation='softmax')(output)
# Se define el modelo total de la red 
lrc_net = Model(inputs = input_tensor, outputs = output)
#ENTRENAMIENTO DE LA RED CONVOLUCIONAL
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
lrc_net.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print('LOGGISTIC REGRESION CLASSIFIER #####################################################################')
history = lrc_net.fit(datosEntrenamiento,etiquetasEntrenamiento,epochs=epochs,batch_size=256,validation_data=(datosValidacion, etiquetasValidacion))
#LOGGER DATOS DE ENTRENAMIENTO
#logger.savedataTrain(history)
#GUARDAR MODELO DE RED CONVOLUCIONAL
cnn_net = Model(inputs = lrc_net.input, outputs = lrc_net.layers[-4].output)
cnn_net.save(os.path.join(logger.path,test+'.h5'))
#GUARDAR MODELO DE RED CONVOLUCIONAL + LRC
lrc_net.save(os.path.join(logger.path,test+'_LRC.h5'))
#EVALUAR MODELO
#GENERACION OA - Overall Accuracy 
test_loss, OA = lrc_net.evaluate(datosPrueba, etiquetasPrueba)
print('LOGISTIC OA:'+str(OA))
#GENERAR MAPA FINAL DE CLASIFICACIÓN
#GENERACION AA - Average Accuracy 
AA = np.zeros(groundTruth.max()+1)
for j in range(1,groundTruth.max()+1):                                     #QUITAR 1 para incluir datos del fondo
    datosClase, etiquetasClase = preparar.extraerDatosClase2D(ventana,j)   #MUESTRAS DE UNA CLASE                 
    test_loss, AA[j] = lrc_net.evaluate(datosClase, etiquetasClase)        #EVALUAR MODELO PARA LA CLASE
#GENERACION Kappa Coefficient
datosSalida = lrc_net.predict(datosPrueba)
datosSalida = datosSalida.argmax(axis=1)
#Guardar los datos de salida
datosSalida_LRC = datosSalida
etiquetasPrueba = etiquetasPrueba.argmax(axis=1)
kappa = cohen_kappa_score(etiquetasPrueba, datosSalida)
#LOGGER DATOS DE VALIDACIÓN
logger.savedataPerformance(OA, AA, kappa)
logger.close()

#############################################################################################################################################################
################Generar caracteristicicas con los datos de entrada y red INCEPTION entrenada#################################################################
features_tr = cnn_net.predict(datosEntrenamiento)
features_test = cnn_net.predict(datosPrueba)
#############################################################################################################################################################

#############################################################################################################################################################
################################## CLASIFICADOR RIEMANN #####################################################################################################
datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)
#CREAR FICHERO DATA LOGGER 
logger = DataLogger(fileName = dataSet+'_RIE', folder = test, save = True) 
#Crear y entrenar clasificador 
print('RIEMANNIAN CLASSIFIER #####################################################################')
rie_net = riemann_classifier(features_tr, etiquetasEntrenamiento, method='tan')
#Guarda el clasificador entrenado
joblib.dump(rie_net, os.path.join(logger.path,test+'_RIE.pkl')) 
#EVALUAR MODELO
#GENERACION OA - Overall Accuracy 
features_testRIE = reshapeFeaturesRIE(features_test)
datosSalida = rie_net.predict(features_testRIE)
#Guardar los datos de salida
datosSalida_RIE = datosSalida
OA = accuracy(etiquetasPrueba, datosSalida)
print('RIEMANNIAN OA:'+str(OA))
#GENERACION AA - Average Accuracy 
AA = np.zeros(groundTruth.max()+1)
for j in range(1,groundTruth.max()+1):                          
    datosClase, etiquetasClase = preparar.extraerDatosClase2D(ventana,j)       #MUESTRAS DE UNA CLASE                 
    features_class = cnn_net.predict(datosClase)
    features_class = reshapeFeaturesRIE(features_class)                   # SOLO PARA RIEMMANIAN CLASSIFIER
    claseSalida = rie_net.predict(features_class)
    AA[j] = accuracy(etiquetasClase, claseSalida)                         #EVALUAR MODELO PARA LA CLASE
#GENERACION Kappa Coefficient
etiquetasPrueba = etiquetasPrueba.argmax(axis=1)
kappa = cohen_kappa_score(etiquetasPrueba, datosSalida)
#LOGGER DATOS DE VALIDACIÓN
logger.savedataPerformance(OA, AA, kappa)
logger.close()
#############################################################################################################################################################
################################## CLASIFICADOR SVM #########################################################################################################
datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)
#CREAR FICHERO DATA LOGGER 
logger = DataLogger(fileName = dataSet+'_SVM', folder = test, save = True) 
#Crear y entrenar clasificador 
print('SVM CLASSIFIER ##########################################################################')
svm_net = svm_classifier(features_tr, etiquetasEntrenamiento, kernel='linear')
#Guarda el clasificador entrenado
joblib.dump(svm_net, os.path.join(logger.path,test+'_SVM.pkl')) 
#EVALUAR MODELO
#GENERACION OA - Overall Accuracy 
features_testSVM = reshapeFeaturesSVM(features_test)
datosSalida = svm_net.predict(features_testSVM)
#Guardar los datos de salida 
datosSalida_SVM = datosSalida  
OA = accuracy(etiquetasPrueba, datosSalida)
print('SVM OA:'+str(OA))
#GENERACION AA - Average Accuracy 
AA = np.zeros(groundTruth.max()+1)
for j in range(1,groundTruth.max()+1):                          
    datosClase, etiquetasClase = preparar.extraerDatosClase2D(ventana,j)       #MUESTRAS DE UNA CLASE                 
    features_class = cnn_net.predict(datosClase)
    features_class = reshapeFeaturesSVM(features_class)                   # SOLO PARA RIEMMANIAN CLASSIFIER
    claseSalida = svm_net.predict(features_class)
    AA[j] = accuracy(etiquetasClase, claseSalida)                         #EVALUAR MODELO PARA LA CLASE
#GENERACION Kappa Coefficient
etiquetasPrueba = etiquetasPrueba.argmax(axis=1)
kappa = cohen_kappa_score(etiquetasPrueba, datosSalida)
#LOGGER DATOS DE VALIDACIÓN
logger.savedataPerformance(OA, AA, kappa)
logger.close()
#############################################################################################################################################################
########################TERMINAR PROCESOS DE KERAS###########################################################################################################
K.clear_session()
#############################################################################################################################################################
#######################RETORNAR ETIQUETAS DE PRUEBA Y PREDICCIÓN#############################################################################################
class_names = []
for i in range(1,groundTruth.max()+1):
    class_names.append('Class '+str(i))
if etiquetasPrueba.ndim>1:
    etiquetasPrueba = etiquetasPrueba.argmax(axis=1)
if datosSalida_LRC.ndim>1:
    datosSalida_LRC = datosSalida_LRC.argmax(axis=1)
if datosSalida_RIE.ndim>1:
    datosSalida_RIE = datosSalida_RIE.argmax(axis=1)
if datosSalida_SVM.ndim>1:
    datosSalida_SVM = datosSalida_SVM.argmax(axis=1)

#########################GENERAR MAPA FINAL DE CLASIFICACIÓN########################
imagenSalida_LRC = preparar.predictionToImage(datosSalida_LRC)
imagenSalida_RIE = preparar.predictionToImage(datosSalida_RIE)
imagenSalida_SVM = preparar.predictionToImage(datosSalida_SVM)
##############################IMAGEN DE COMPARACION#################################
imgCompare_LRC = np.absolute(groundTruth-imagenSalida_LRC)
imgCompare_LRC = (imgCompare_LRC > 0) * 1 
imgCompare_RIE = np.absolute(groundTruth-imagenSalida_RIE)
imgCompare_RIE = (imgCompare_RIE > 0) * 1 
imgCompare_SVM = np.absolute(groundTruth-imagenSalida_SVM)
imgCompare_SVM = (imgCompare_SVM > 0) * 1 
###############################PRESENTAR IMAGENES DE SALIDA#########################
data.graficarHsi_VS(groundTruth, imagenSalida_LRC)
data.graficarHsi_VS(groundTruth, imgCompare_LRC,cmap ='bw')
data.graficarHsi_VS(groundTruth, imagenSalida_RIE)
data.graficarHsi_VS(groundTruth, imgCompare_RIE,cmap ='bw')
data.graficarHsi_VS(groundTruth, imagenSalida_SVM)
data.graficarHsi_VS(groundTruth, imgCompare_SVM,cmap ='bw')
