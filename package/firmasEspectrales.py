#Permite extraer conjuntos de entrenamiento, validacion y prueba en 1D y 2D
#Trabaja con la informacion del fondo (clase cero) o la descarta para el procesamiento de los datos.
import math
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

class FirmasEspectrales:
    
    def __init__(self, dataImagen, groundTruth, backGround = True):
        self.dataImagen = dataImagen
        self.groundTruth = groundTruth
        self.indices_org = []
        
        if backGround == True:
            for i in range(dataImagen.shape[1]):
                for j in range(dataImagen.shape[2]):
                    self.indices_org.append([i, j]) 
            self.indices_org = np.array(self.indices_org, dtype=np.int16).T
            self.indices = self.indices_org.copy()
            np.random.shuffle(self.indices.T)  

        if backGround == False:
            for i in range(dataImagen.shape[1]):
                for j in range(dataImagen.shape[2]):
                    if self.groundTruth[i,j] != 0:
                        self.indices_org.append([i, j]) 
            self.indices_org = np.array(self.indices_org, dtype=np.int16).T
            self.indices = self.indices_org.copy()
            np.random.shuffle(self.indices.T) 
        
    def __str__(self):
        return f"Dimensiones Imagen:\t {self.dataImagen.shape}\n" \
               f"Dimensiones Ground Truth:\t {self.groundTruth.shape}\n" \
               f"Dimensiones Vector Indices:\t {self.indices.shape}\n"