#PRUEBA DE REDES PROFUNDAS.
# Se crea el archivo xxx_TEST.txt y se carga las redes entrenadas para econtrar: 
    #Overall Accuracy 
    #Average Accuracy 
    #Kappa Coefficient
# Cambiar cargar datos 2D o 3D dependiendo de la prueba realizada

from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from package.MorphologicalProfiles import morphologicalProfiles
from package.dataLogger import DataLogger

import matplotlib.pyplot as plt
import numpy as np

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Conv2DTranspose, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Flatten
from keras.models import Model
from keras.optimizers import SGD
from keras import regularizers
from keras.models import load_model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, cohen_kappa_score

