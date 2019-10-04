#FEATURE EXTRACTION AND CLASSIFIER
# Utiliza las caracteristica generadas por los esquemas de estracción de caracteristicas no supervisados
# implementa una capa de salida supervisada.
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

from pyriemann.estimation import Covariances
from pyriemann.classification import MDM
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def lr_classifier(features_tr, features_val, etiquetasEntrenamiento, etiquetasValidacion):
    input_features = Input(shape=(features_tr.shape[1],features_tr.shape[2],features_tr.shape[3]))
    fullconected = Flatten()(input_features)
    fullconected = Dense(128, activation = 'relu')(fullconected) 
    fullconected = Dense(groundTruth.max()+1, activation = 'softmax')(fullconected) 
    classifier = Model(inputs = input_features, outputs = fullconected) #GENERA EL MODELO FINAL

    print(classifier.summary())
    classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])   
    history = classifier.fit(features_tr, etiquetasEntrenamiento, epochs=epochs, batch_size=128, shuffle=True, validation_data=(features_val, etiquetasValidacion))
    return classifier, history

def svm_classifier(features_tr, etiquetasEntrenamiento, features_test, kernel='linear'):
    #Reshape features (n_samples, m_features)
    features_tr = features_tr.reshape(features_tr.shape[0],features_tr.shape[1]*features_tr.shape[2],features_tr.shape[3])
    features_tr = features_tr.reshape(features_tr.shape[0],features_tr.shape[1]*features_tr.shape[2])
    features_test = features_test.reshape(features_test.shape[0],features_test.shape[1]*features_test.shape[2],features_test.shape[3])
    features_test = features_test.reshape(features_test.shape[0],features_test.shape[1]*features_test.shape[2])
    #Reshape labels from categorical 
    etiquetasEntrenamiento = np.argmax(etiquetasEntrenamiento, axis=1)
    #SVM Classifier one-against-one
    classifier = svm.SVC(gamma='scale', decision_function_shape='ovo', kernel=kernel, verbose=True)
    classifier.fit(features_tr,etiquetasEntrenamiento)
    return classifier, features_test

def riemann_classifier(features_tr, etiquetasEntrenamiento, features_test, method='tan'):
    #Reshape features (n_samples, m_filters, p_features)
    features_tr = features_tr.reshape(features_tr.shape[0],features_tr.shape[1]*features_tr.shape[2],features_tr.shape[3])
    features_tr = np.transpose(features_tr, (0, 2, 1))
    features_test = features_test.reshape(features_test.shape[0],features_test.shape[1]*features_test.shape[2],features_test.shape[3])
    features_test = np.transpose(features_test, (0, 2, 1))
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
    return classifier, features_test

def accuracy(y_true, y_pred):
    if y_true.ndim>1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim>1:
        y_pred = np.argmax(y_pred, axis=1)

    OA = accuracy_score(y_true, y_pred)
    print(OA)

#CARGAR IMAGEN HSI Y GROUND TRUTH
numTest = 1
dataSet = 'PaviaU'
ventana = 8 #VENTANA 2D de PROCESAMIENTO
data = CargarHsi(dataSet)
imagen = data.imagen
groundTruth = data.groundTruth

#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis()
imagenPCA = pca.pca_calculate(imagen, varianza=0.95)
#imagenPCA = pca.pca_calculate(imagen, componentes=4)
print(imagenPCA.shape)

#ESTIMACIÓN DE EXTENDED EXTINTION PROFILES
mp = morphologicalProfiles()
imagenEEP = mp.EEP(imagenPCA, num_levels=6)    
print(imagenEEP.shape)

#PREPARAR DATOS PARA ENTRENAMIENTO
preparar = PrepararDatos(imagenEEP, groundTruth, False)
datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,20,ventana)
datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)

######################LOAD STACKED CONVOLUTIONAL AUTOENCODER#####################################################
epochs = 50 #número de iteraciones
encoder = load_model('FE_BCAE02.h5', custom_objects={'euclidean_distance_loss': euclidean_distance_loss})   
print(encoder.summary()) 

#Generar caracteristicicas con los datos de entrada
features_tr = encoder.predict(datosEntrenamiento)
features_val = encoder.predict(datosValidacion)
features_test = encoder.predict(datosPrueba)
history = 0

################################## CLASIFICADOR RIEMANN #########################################################
classifier, features_test = riemann_classifier(features_tr, etiquetasEntrenamiento, features_test,  method='tan')
#################################################################################################################

################################## CLASIFICADOR SVM #############################################################
#classifier, features_test = svm_classifier(features_tr, etiquetasEntrenamiento, features_test, kernel='linear')
#################################################################################################################

##################################CLASIFICADOR LOGISTIC REGRESSION###############################################
#classifier, history = lr_classifier(features_tr, features_val, etiquetasEntrenamiento, etiquetasValidacion)
#################################################################################################################

############################## GRAFICAS Y SALIDAS ###############################################################
datosSalida = classifier.predict(features_test)
imagenSalida = preparar.predictionToImage(datosSalida)
data.graficarHsi_VS(groundTruth, imagenSalida)
accuracy(etiquetasPrueba, datosSalida)
data.graficar_history(history)
K.clear_session()

