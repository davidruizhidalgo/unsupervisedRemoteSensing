#Permite extraer conjuntos de entrenamiento, validacion y prueba en 1D y 2D
#OMITE LA INFOMACION DEL FONDO
import math
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

class PrepararDatosSinFondo:
    
    def __init__(self, dataImagen, groundTruth):
        self.dataImagen = dataImagen
        self.groundTruth = groundTruth
        self.indices_org = []
        
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
               f"Dimensiones Vector Indices Originales:\t {self.indices_org.shape}\n" \
               f"Dimensiones Vector Indices Shuffle:\t {self.indices.shape}\n"

    def extraerDatos1D(self, porEntrenamiento, porValidacion):
        #Datos de entrenamiento
        numtrain = self.indices.shape[1]*porEntrenamiento/100
        numtrain = math.floor(numtrain)
        datosEntrenamiento = np.zeros( (numtrain,self.dataImagen.shape[0]) )
        etiquetasEntrenamiento =  np.zeros( numtrain )
        
        for i in range(numtrain):
            datosEntrenamiento[i,:] = self.dataImagen[:,self.indices[0,i],self.indices[1,i]]
            etiquetasEntrenamiento[i] = self.groundTruth[self.indices[0,i],self.indices[1,i]]

        #Datos de validación
        numVal = self.indices.shape[1]*porValidacion/100
        numVal = math.floor(numVal)
        datosValidacion = np.zeros( (numVal,self.dataImagen.shape[0]) )
        etiquetasValidacion =  np.zeros( numVal )

        for i in range(numVal):
            datosValidacion[i,:] = self.dataImagen[:,self.indices[0,numtrain+i],self.indices[1,numtrain+i]]
            etiquetasValidacion[i] = self.groundTruth[self.indices[0,numtrain+i],self.indices[1,numtrain+i]]
        
        #Codificar etiquetas de entrenamiento y validacion ONE HOT
        etiquetasEntrenamiento = to_categorical(etiquetasEntrenamiento)
        etiquetasValidacion = to_categorical(etiquetasValidacion)     

        return datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion

    def extraerDatos2D(self, porEntrenamiento, porValidacion, ventana):
        pass