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
###########################PROGRAMA PRINCIPAL################################################################################################################
dataSet = 'IndianPines'
test = 'eapInception'   # pcaInception eapInception
save = True             # false to avoid create logger
fe_eap = True           # false for PCA, true for EAP 

data = CargarHsi(dataSet)
imagen = data.imagen
groundTruth = data.groundTruth
print(imagen.shape)

########################CREAR FICHEROS DATA LOGGER########################################################################################################### 
logger_LRC = DataLogger(fileName = dataSet+'_LRC', folder = test, save = save)  
path = logger_LRC.path
print(path)
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

data.graficarHsi_VS(groundTruth, imagenFE[0])


