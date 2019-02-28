#Permite extraer conjuntos de entrenamiento, validacion y prueba en 1D y 2D
import math
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

class PrepararDatos:
    
    def __init__(self,dataImagen, groundTruth):
        self.dataImagen = dataImagen
        self.groundTruth = groundTruth
        self.indices = np.zeros( (2,dataImagen.shape[1]*dataImagen.shape[2]), dtype=np.int16 )
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

    def extraerDatos1D(self,porEntrenamiento, porValidacion):
        #Datos de entrenamiento
        numtrain = self.dataImagen.shape[1]*self.dataImagen.shape[2]*porEntrenamiento/100
        numtrain = math.floor(numtrain)
        datosEntrenamiento = np.zeros( (numtrain,self.dataImagen.shape[0]) )
        etiquetasEntrenamiento =  np.zeros( numtrain )
        
        for i in range(numtrain):
            datosEntrenamiento[i,:] = self.dataImagen[:,self.indices[0,i],self.indices[1,i]]
            etiquetasEntrenamiento[i] = self.groundTruth[self.indices[0,i],self.indices[1,i]]

        #Datos de validación
        numVal = self.dataImagen.shape[1]*self.dataImagen.shape[2]*porValidacion/100
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
    
    def extraerDatos2D(self,porEntrenamiento, porValidacion):
        #Datos de entrenamiento
        numtrain = self.dataImagen.shape[1]*self.dataImagen.shape[2]*porEntrenamiento/100
        numtrain = math.floor(numtrain)
        datosEntrenamiento = np.zeros( (numtrain,3,3,self.dataImagen.shape[0]) )
        etiquetasEntrenamiento =  np.zeros( numtrain )
        paddingImage = np.zeros((self.dataImagen.shape[0],self.dataImagen.shape[1]+2,self.dataImagen.shape[2]+2))
        paddingImage[:,1:-1,1:-1] =  self.dataImagen #imagen aunmentada para padding
        for i in range(numtrain):
          datosEntrenamiento[i] = paddingImage[:,self.indices[0,i]:self.indices[0,i]+3,self.indices[1,i]:self.indices[1,i]+3].T
          etiquetasEntrenamiento[i] = self.groundTruth[self.indices[0,i],self.indices[1,i]]

        #Datos de validación
        numVal = self.dataImagen.shape[1]*self.dataImagen.shape[2]*porValidacion/100
        numVal = math.floor(numVal)
        datosValidacion = np.zeros( (numVal,3,3,self.dataImagen.shape[0]) )
        etiquetasValidacion =  np.zeros( numVal )
        for i in range(numVal):
            datosValidacion[i] = paddingImage[:,self.indices[0,numtrain+i]:self.indices[0,numtrain+i]+3,self.indices[1,numtrain+i]:self.indices[1,numtrain+i]+3].T
            etiquetasValidacion[i] = self.groundTruth[self.indices[0,numtrain+i],self.indices[1,numtrain+i]]

        #Codificar etiquetas de entrenamiento y validacion ONE HOT
        etiquetasEntrenamiento = to_categorical(etiquetasEntrenamiento)
        etiquetasValidacion = to_categorical(etiquetasValidacion)        

        return datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion

    def extraerDatosPrueba1D(self):
        k=0 #Generacion de vector de indices de datos en la imagen HSI
        indices = np.zeros( (2,self.dataImagen.shape[1]*self.dataImagen.shape[2]), dtype=np.int16 )
        for i in range(self.dataImagen.shape[1]):
            for j in range(self.dataImagen.shape[2]):
                indices[:,k] = [i, j]
                k += 1
        #Datos de Prueba
        numprueba = self.dataImagen.shape[1]*self.dataImagen.shape[2]
        datosPrueba = np.zeros( (numprueba,self.dataImagen.shape[0]) )
        etiquetasPrueba =  np.zeros( numprueba )
        
        for i in range(numprueba):
            datosPrueba[i,:] = self.dataImagen[:,self.indices[0,i],self.indices[1,i]]
            etiquetasPrueba[i] = self.groundTruth[self.indices[0,i],self.indices[1,i]]

        return datosPrueba, etiquetasPrueba

    def extraerDatosPrueba2D(self):
        k=0 #Generacion de vector de indices de datos en la imagen HSI
        indices = np.zeros( (2,self.dataImagen.shape[1]*self.dataImagen.shape[2]), dtype=np.int16 )
        for i in range(self.dataImagen.shape[1]):
            for j in range(self.dataImagen.shape[2]):
                indices[:,k] = [i, j]
                k += 1
        #Datos de Prueba
        numData = self.dataImagen.shape[1]*self.dataImagen.shape[2]
        datosPrueba = np.zeros( (numData,3,3,self.dataImagen.shape[0]) )
        etiquetasPrueba =  np.zeros( numData )
        paddingImage = np.zeros((self.dataImagen.shape[0],self.dataImagen.shape[1]+2,self.dataImagen.shape[2]+2))
        paddingImage[:,1:-1,1:-1] =  self.dataImagen #imagen aunmentada para padding
        for i in range(numData):
          datosPrueba[i] = paddingImage[:,self.indices[0,i]:self.indices[0,i]+3,self.indices[1,i]:self.indices[1,i]+3].T
          etiquetasPrueba[i] = self.groundTruth[self.indices[0,i],self.indices[1,i]]

        return datosPrueba, etiquetasPrueba