#PRUEBA DE REDES PROFUNDAS SEMISUPERVISADAS.
# Se crea el archivo xxx_TEST.txt y se cargan las redes entrenadas para econtrar: 
    #Overall Accuracy 
    #Average Accuracy 
    #Kappa Coefficient

from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from package.MorphologicalProfiles import morphologicalProfiles
from package.dataLogger import DataLogger

import matplotlib.pyplot as plt
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
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score, cohen_kappa_score

