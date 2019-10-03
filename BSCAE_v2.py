#BRANCHES OF CONVOLUTIONAL STACKED AUTOENCODERS V2 - INCEPTION BASED  
#Se implementa el encoder y el decoder convolucional utilizando la arquitectura de la red INCEPTION y una capa
# de recosntrucción tipo refinement_layer para aprender una representación de las caracteristicas de los datos de entrada.

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

###############################FUNCIONES########################################################################################################################
def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))

###########################PROGRAMA PRINCIPAL################################################################################################################