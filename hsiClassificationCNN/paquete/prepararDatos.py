#Permite extraer conjuntos de entrenamiento, validacion y prueba

import numpy as np
import matplotlib.pyplot as plt

class PrepararDatos:
    
    def __init__(self,dataImagen, groundTruth):
        self.dataImagen = dataImagen
        self.groundTruth = groundTruth
        self.indices = np.zeros( (2,dataImagen.shape[1]*dataImagen.shape[2]) )
        k=0 #Generacion de vector de indices de datos en la imagen HSI
        for i in range(dataImagen.shape[1]):
            for j in range(dataImagen.shape[2]):
                self.indices[:,k] = [i, j]
                k += 1
        np.random.shuffle(self.indices.T)     

    def __str__(self):
        return f"Dimensiones Imagen:\t {self.dataImagen.shape}\n" \
               f"Dimensiones Ground Truth:\t {self.groundTruth.shape}\n" \
               f"Dimensiones Vector Indices:\t {self.indices.shape}\n"

    def extraerDatos1D(self):
        #Datos de entrenamiento
        datosEntrenamiento = self.dataImagen[:,0,0]
        etiquetasEntrenamiento = [0,1,2,3]

        datosValidacion = [0,1,2,3]
        etiquetasValidacion = [0,1,2,3]

        return datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion
    
    def funcionPrueba(self):
        a = np.array([ [1,1,1,2,2,2,3,3,3],[1,2,3,4,5,6,7,8,9]])
        print(a)
        print(a.shape)
        np.random.shuffle(a.T)
        print(a)
        print(a.shape)