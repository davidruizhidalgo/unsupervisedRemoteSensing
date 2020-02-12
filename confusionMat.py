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

def reshapeFeaturesSVM(features_test):
    features_test = features_test.reshape(features_test.shape[0],features_test.shape[1]*features_test.shape[2],features_test.shape[3])
    features_test = features_test.reshape(features_test.shape[0],features_test.shape[1]*features_test.shape[2])
    return features_test

def reshapeFeaturesRIE(features_test):
    features_test = features_test.reshape(features_test.shape[0],features_test.shape[1]*features_test.shape[2],features_test.shape[3])
    features_test = np.transpose(features_test, (0, 2, 1))
    return features_test

###########################PROGRAMA PRINCIPAL################################################################################################################
###########################CARGAR IMAGEN HSI, GROUND TRUTH e INICIAR LOGGER##################################################################################
numTest = 10
dataSet = 'IndianPines'
test = 'eapInception'    # pcaInception eapInception
save = False             # false to avoid create logger
fe_eap = True            # false for PCA, true for EAP 
vectNets =[0, 0, 0, 0]   # [FE, LRC, RIE, SVM]   

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

########################ENLAZAR FICHEROS DATA LOGGER########################################################################################################### 
logger_LRC = DataLogger(fileName = dataSet+'_LRC', folder = test, save = save)   
logger_RIE = DataLogger(fileName = dataSet+'_RIE', folder = test, save = save) 
logger_SVM = DataLogger(fileName = dataSet+'_SVM', folder = test, save = save) 
########################PREPARAR DATOS PARA VALIDACION#########################################################################################################
ventana = 9  #VENTANA 2D de PROCESAMIENTO
preparar = PrepararDatos(imagenFE, groundTruth, False)
datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)    
########################CARGAR FEATURE EXTRACTION NETWORK######################################################################################################
feNet = load_model(os.path.join(logger_LRC.path,test+str(vectNets[0])+'.h5'))
#Generar caracteristicas con la FE Net
features_test = feNet.predict(datosPrueba)
########################LOGISTIC REGRESSION CLASSIFIER#########################################################################################################
lrcNet = load_model(os.path.join(logger_LRC.path,test+str(vectNets[1])+'_LRC.h5'))
lrcSalida = lrcNet.predict(datosPrueba)
print('LOGISTIC REGRESSION CLASSIFIER DONE!!!!!!!!!!!!!!!!!!!!!!!!!')
########################RIEMANNIAN  CLASSIFIER#################################################################################################################
rieNet = joblib.load(os.path.join(logger_RIE.path,test+'_RIE'+str(vectNets[2])+'.pkl')) 
features_rie = reshapeFeaturesRIE(features_test)
rieSalida = rieNet.predict(features_rie)
print('RIEMANNIAN CLASSIFIER DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
########################SUPPORT VECTOR MACHINE CLASSIFIER######################################################################################################
svmNet = joblib.load(os.path.join(logger_SVM.path,test+'_SVM'+str(vectNets[2])+'.pkl')) 
features_svm = reshapeFeaturesSVM(features_test)
svmSalida = svmNet.predict(features_svm)
print('SUPPORT VECTOR MACHINE  DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
########################AJUSTAR DATOS DE SALIDA################################################################################################################
class_names = []
for i in range(1,groundTruth.max()+1):
    class_names.append('Class '+str(i))
if etiquetasPrueba.ndim>1:
    etiquetasPrueba = etiquetasPrueba.argmax(axis=1)
if lrcSalida.ndim>1:
    lrcSalida = lrcSalida.argmax(axis=1)
if rieSalida.ndim>1:
    rieSalida = rieSalida.argmax(axis=1)
if svmSalida.ndim>1:
    svmSalida = svmSalida.argmax(axis=1)
########################GENERAR MATRIZ DE CONFUSION############################################################################################################
fig_lrc = plotlyConfusionMatrix(true_labels=etiquetasPrueba, pred_labels=lrcSalida, class_names=class_names)
fig_lrc.write_image(os.path.join(logger_LRC.path,'confusionMat_'+str(vectNets[1])+'.png'))

fig_rie = plotlyConfusionMatrix(true_labels=etiquetasPrueba, pred_labels=rieSalida, class_names=class_names)
fig_rie.write_image(os.path.join(logger_RIE.path,'confusionMat_'+str(vectNets[2])+'.png'))

fig_svm = plotlyConfusionMatrix(true_labels=etiquetasPrueba, pred_labels=svmSalida, class_names=class_names)
fig_svm.write_image(os.path.join(logger_SVM.path,'confusionMat_'+str(vectNets[3])+'.png'))
print('CONFUSION MATRIX DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
###############################################################################################################################################################