#Script utilizado para el desarrollo de pruebas en el codigo. 
####### EN DESARROLLO ##########################
# 1. Desarrollar una arquitectura de red selección de caracteristicas NO SUPERVISADA - Capa de Salida FINE-TUNNING
        # CNN Autoendoder and full conected network +/- 75% acc
            #Desempeño con EEP:  +/- 85% acc
            #SVM lineal and rbf
        # Grafo de CNN autoencoders
        # Evaluar funciones modificaciones en la funcion de costo
        # Evaluar estrategias de concatenacion en BSCAE

# 2. Desarrollar un esquema de data augmentation => prepararDatos.py
# 4. Revisar documentacion reciente del estado del arte.

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
