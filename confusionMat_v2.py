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
###########################PROGRAMA PRINCIPAL################################################################################################################
###########################CARGAR IMAGEN HSI, GROUND TRUTH e INICIAR LOGGER##################################################################################
dataSet = 'KSC'
test =  'BCAE'          # test folder
fe_eap = True            # false for PCA, true for EAP 
vectNets =[5, 5]          # [FE, LRC]   

data = CargarHsi(dataSet)
imagen = data.imagen
groundTruth = data.groundTruth
print(imagen.shape)
###########################INICIAL FEATURE EXTRACTION########################################################################################################
#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis()
imagenFE = pca.pca_calculate(imagen, varianza=0.95)
#imagenFE = pca.pca_calculate(imagen, componentes=18)
print(imagenFE.shape)

#ESTIMACIÓN DE EXTENDED EXTINTION PROFILES
if fe_eap:    
    mp = morphologicalProfiles()
    imagenFE = mp.EEP(imagenFE, num_levels=4)    
    print(imagenFE.shape)

########################ENLAZAR FICHEROS DATA LOGGER########################################################################################################### 
logger_LRC = DataLogger(fileName = dataSet, folder = test, save = False) 
########################PREPARAR DATOS PARA VALIDACION#########################################################################################################
ventana = 8  #VENTANA 2D de PROCESAMIENTO
preparar = PrepararDatos(imagenFE, groundTruth, False)
datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)    
########################CARGAR FEATURE EXTRACTION NETWORK######################################################################################################
feNet = load_model(os.path.join(logger_LRC.path,'FE_'+test+str(vectNets[0])+'.h5'), custom_objects={'euclidean_distance_loss': euclidean_distance_loss}) 
#Generar caracteristicas con la FE Net
features_test = feNet.predict(datosPrueba)

########################LOGISTIC REGRESSION CLASSIFIER#########################################################################################################
lrcNet = load_model(os.path.join(logger_LRC.path,'C_'+test+str(vectNets[1])+'.h5'))
lrcSalida = lrcNet.predict(features_test)
print('LOGISTIC REGRESSION CLASSIFIER DONE!!!!!!!!!!!!!!!!!!!!!!!!!')
####################################GENERAR MAPA FINAL DE CLASIFICACIÓN#####################################################################
imagenSalida = preparar.predictionToImage(lrcSalida)
data.graficarHsi_VS(groundTruth, imagenSalida)
#COMPARACION
dataCompare = np.absolute(groundTruth-imagenSalida)
dataCompare = (dataCompare > 0) * 1 
data.graficarHsi_VS(imagenSalida, dataCompare, cmap ='bw')
########################AJUSTAR DATOS DE SALIDA################################################################################################################
class_names = []
for i in range(1,groundTruth.max()+1):
    class_names.append('Class '+str(i))
if etiquetasPrueba.ndim>1:
    etiquetasPrueba = etiquetasPrueba.argmax(axis=1)
if lrcSalida.ndim>1:
    lrcSalida = lrcSalida.argmax(axis=1)
########################GENERAR MATRIZ DE CONFUSION############################################################################################################
fig_lrc = plotlyConfusionMatrix(true_labels=etiquetasPrueba, pred_labels=lrcSalida, class_names=class_names)
fig_lrc.write_image(os.path.join(logger_LRC.path,'confusionMat_'+str(vectNets[1])+'.png'))
K.clear_session()
print('CONFUSION MATRIX DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
###############################################################################################################################################################