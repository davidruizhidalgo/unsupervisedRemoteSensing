#script para cargar clasificadores entrenados y generar la matriz de confusión
# pylint: disable=E1136  # pylint/issues/3139
import warnings
warnings.filterwarnings('ignore')
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
from keras.models import load_model
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

import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import download_plotlyjs, plot
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from scipy.stats import binom

###########################FUNCIONES#########################################################################################################################
def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

def plotlyConfusionMatrix(true_labels, pred_labels, class_names): 
  cm = confusion_matrix(true_labels, pred_labels).astype(float)
  sum_hor = cm.sum(axis=1)
  
  accuracy = 0
  for i in range(len(sum_hor)):
    accuracy = accuracy + cm[i,i]
  accuracy = 100.0*accuracy/sum_hor.sum()
  
  for i in range(len(sum_hor)):
    cm[i,:] = (100.0/sum_hor[i])*cm[i,:]
  cm = np.flip(cm,axis=0)
  
  alpha = 0.05
  n = len(true_labels)
  c = len(class_names)
  chancelevel = (100.0/n)*binom.ppf(1.0-alpha, n, 1.0/c)
  
  y=[name+' ('+str(val)+')' for name,val in zip(class_names[::-1],sum_hor)]
  data=go.Heatmap(
                  z=cm.tolist(),
                  x=class_names,
                  y=y,
                  colorscale='Rainbow'
       )
	   #- Chance level = '+str(chancelevel)+' %'
  layout = go.Layout(
                  title='Confusion Matrix (z in %) - Overall Accuracy = '+
                        str(accuracy)+' % ',

                  xaxis=dict(
                          title='Predicted Labels'
                  ),
                  yaxis=dict(
                          title='True Labels',
                  )
          )
  fig = go.Figure(data=data, layout=layout)
  return fig

def reshapeFeatures(features_test, method='RIEM'):
    if method == 'RIEM':
        features_test = features_test.reshape(features_test.shape[0],features_test.shape[1]*features_test.shape[2],features_test.shape[3])
        features_test = np.transpose(features_test, (0, 2, 1))
    else: 
        features_test = features_test.reshape(features_test.shape[0],features_test.shape[1]*features_test.shape[2],features_test.shape[3])
        features_test = features_test.reshape(features_test.shape[0],features_test.shape[1]*features_test.shape[2])
    return features_test

###########################PROGRAMA PRINCIPAL################################################################################################################
###########################CARGAR IMAGEN HSI, GROUND TRUTH e INICIAR LOGGER##################################################################################
dataSet = 'KSC'
test =    'BCAE'          # test folder
algorithm = 'RIEM'        # LRCL, SVMC, RIEM
fe_eep = True             # false for PCA, true for EEP 
vectNets = [1, 1, 1, 1]    # [FE, LRCL, SVMC, RIEM]   

data = CargarHsi(dataSet)
imagen = data.imagen
groundTruth = data.groundTruth
print(imagen.shape)
###########################INICIAL FEATURE EXTRACTION#########################################################################################################
#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis()
imagenFE = pca.pca_calculate(imagen, varianza=0.95)
print(imagenFE.shape)
##########################EXTENDED EXTINTION PROFILES##########################################################################################################
if fe_eep:    
    mp = morphologicalProfiles()
    imagenFE = mp.EEP(imagenFE, num_levels=4)    
    print(imagenFE.shape)
########################ENLAZAR FICHEROS DATA LOGGER########################################################################################################### 
logger = DataLogger(fileName = dataSet, folder = test, save = False) 
########################PREPARAR DATOS PARA VALIDACION#########################################################################################################
ventana = 8  #VENTANA 2D de PROCESAMIENTO
preparar = PrepararDatos(imagenFE, groundTruth, False)
datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)    
########################CARGAR FEATURE EXTRACTION NETWORK######################################################################################################
feNet = load_model(os.path.join(logger.path,'FE_'+test+str(vectNets[0])+'.h5'), custom_objects={'euclidean_distance_loss': euclidean_distance_loss}) 
#Generar caracteristicas con la FE Net
features_test = feNet.predict(datosPrueba)

############################### CLASSIFIER ALGORITHM############################################################################################################
########################LOGISTIC REGRESSION CLASSIFIER##########################################################################################################
if algorithm == 'LRCL':
  classifier = load_model(os.path.join(logger.path,'C_'+test+str(vectNets[1])+'.h5')) 
  print('LOGISTIC REGRESSION CLASSIFIER DONE!!!!!!!!!!!!!!!!!!!!!!!!!')
#########################RIEMANIAN CLASSIFIER####################################################################################################################
if algorithm == 'SVMC': 
  classifier = joblib.load(os.path.join(logger.path,test+'_'+algorithm+str(vectNets[2])+'.pkl')) 
  features_test = reshapeFeatures(features_test, algorithm)
#########################SVM CLASSIFIER##########################################################################################################################
if algorithm == 'RIEM': 
  classifier = joblib.load(os.path.join(logger.path,test+'_'+algorithm+str(vectNets[3])+'.pkl')) 
  features_test = reshapeFeatures(features_test, algorithm)
#################################################################################################################################################################
####################################GENERAR MAPA FINAL DE CLASIFICACIÓN##########################################################################################
datosSalida = classifier.predict(features_test)
imagenSalida = preparar.predictionToImage(datosSalida)
data.graficarHsi_VS(groundTruth, imagenSalida)
##############################IMAGEN DE COMPARACION##############################################################################################################
dataCompare = np.absolute(groundTruth-imagenSalida)
dataCompare = (dataCompare > 0) * 1 
data.graficarHsi_VS(imagenSalida, dataCompare, cmap ='bw')
########################AJUSTAR DATOS DE SALIDA#################################################################################################################
class_names = []
for i in range(1,groundTruth.max()+1):
    class_names.append('Class '+str(i))
if etiquetasPrueba.ndim>1:
    etiquetasPrueba = etiquetasPrueba.argmax(axis=1)
if datosSalida.ndim>1:
    datosSalida = datosSalida.argmax(axis=1)
########################GENERAR MATRIZ DE CONFUSION############################################################################################################
fig_lrc = plotlyConfusionMatrix(true_labels=etiquetasPrueba, pred_labels=datosSalida, class_names=class_names)
fig_lrc.write_image(os.path.join(logger.path,'confusionMat_'+algorithm+str(vectNets[1])+'.png'))
K.clear_session()
print('CONFUSION MATRIX DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
###############################################################################################################################################################