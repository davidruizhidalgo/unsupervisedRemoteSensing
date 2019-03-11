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
    
    def extraerDatos2D(self,porEntrenamiento, porValidacion, ventana):
        #Datos de entrenamiento
        numtrain = self.indices.shape[1]*porEntrenamiento/100
        numtrain = math.floor(numtrain)
        datosEntrenamiento = np.zeros( (numtrain,ventana,ventana,self.dataImagen.shape[0]) )
        etiquetasEntrenamiento =  np.zeros( numtrain )
        paddingImage = np.zeros((self.dataImagen.shape[0],self.dataImagen.shape[1]+ventana-1,self.dataImagen.shape[2]+ventana-1))
        paddingImage[:,math.floor((ventana-1)/2):self.dataImagen.shape[1]+math.floor((ventana-1)/2),math.floor((ventana-1)/2):self.dataImagen.shape[2]+math.floor((ventana-1)/2)] =  self.dataImagen #imagen aunmentada para padding
        for i in range(numtrain):
          datosEntrenamiento[i] = paddingImage[:,self.indices[0,i]:self.indices[0,i]+ventana,self.indices[1,i]:self.indices[1,i]+ventana].T
          etiquetasEntrenamiento[i] = self.groundTruth[self.indices[0,i],self.indices[1,i]]

        #Datos de validación
        numVal = self.indices.shape[1]*porValidacion/100
        numVal = math.floor(numVal)
        datosValidacion = np.zeros( (numVal,ventana,ventana,self.dataImagen.shape[0]) )
        etiquetasValidacion =  np.zeros( numVal )
        for i in range(numVal):
            datosValidacion[i] = paddingImage[:,self.indices[0,numtrain+i]:self.indices[0,numtrain+i]+ventana,self.indices[1,numtrain+i]:self.indices[1,numtrain+i]+ventana].T
            etiquetasValidacion[i] = self.groundTruth[self.indices[0,numtrain+i],self.indices[1,numtrain+i]]

        #Codificar etiquetas de entrenamiento y validacion ONE HOT
        etiquetasEntrenamiento = to_categorical(etiquetasEntrenamiento)
        etiquetasValidacion = to_categorical(etiquetasValidacion)        

        return datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion

    def extraerDatosPrueba1D(self):
        k=0 #Generacion de vector de indices de datos en la imagen HSI
        indices = np.zeros( (2,self.indices.shape[1]), dtype=np.int16 )
        for i in range(self.dataImagen.shape[1]):
            for j in range(self.dataImagen.shape[2]):
                indices[:,k] = [i, j]
                k += 1
        #Datos de Prueba
        numprueba = self.indices.shape[1]
        datosPrueba = np.zeros( (numprueba,self.dataImagen.shape[0]) )
        etiquetasPrueba =  np.zeros( numprueba )
        
        for i in range(numprueba):
            datosPrueba[i,:] = self.dataImagen[:,indices[0,i],indices[1,i]]
            etiquetasPrueba[i] = self.groundTruth[indices[0,i],indices[1,i]]

        return datosPrueba, etiquetasPrueba

    def extraerDatosPrueba2D(self,ventana):
        k=0 #Generacion de vector de indices de datos en la imagen HSI
        indices = np.zeros( (2,self.indices.shape[1]), dtype=np.int16 )
        for i in range(self.dataImagen.shape[1]):
            for j in range(self.dataImagen.shape[2]):
                indices[:,k] = [i, j]
                k += 1
        #Datos de Prueba
        numData = self.indices.shape[1]
        datosPrueba = np.zeros( (numData,ventana,ventana,self.dataImagen.shape[0]) )
        etiquetasPrueba =  np.zeros( numData )
        paddingImage = np.zeros((self.dataImagen.shape[0],self.dataImagen.shape[1]+ventana-1,self.dataImagen.shape[2]+ventana-1))
        paddingImage[:,math.floor((ventana-1)/2):self.dataImagen.shape[1]+math.floor((ventana-1)/2),math.floor((ventana-1)/2):self.dataImagen.shape[2]+math.floor((ventana-1)/2)] =  self.dataImagen #imagen aunmentada para padding
        for i in range(numData):
          datosPrueba[i] = paddingImage[:,indices[0,i]:indices[0,i]+ventana,indices[1,i]:indices[1,i]+ventana].T
          etiquetasPrueba[i] = self.groundTruth[indices[0,i],indices[1,i]]
        
        #Codificar etiquetas de prueba en ONE HOT
        etiquetasPrueba = to_categorical(etiquetasPrueba)

        return datosPrueba, etiquetasPrueba

    def extraerDatosClase1D(self, clase, numCls):
        datosClase = []
        etiquetasClase = []
        
        for i in range(self.dataImagen.shape[1]):
            for j in range(self.dataImagen.shape[2]):
                if clase == self.groundTruth[i,j]:
                    datosClase.append(self.dataImagen[:,i,j]) 
                    etiquetasClase.append(self.groundTruth[i,j])
        
        datosClase = np.array(datosClase)
        etiquetasClase = np.array(etiquetasClase)
        etiquetasClase = to_categorical(etiquetasClase, num_classes=numCls)
        return datosClase, etiquetasClase

    def extraerDatosClase2D(self, ventana, clase, numCls):
        datosClase = []
        etiquetasClase = []
        paddingImage = np.zeros((self.dataImagen.shape[0],self.dataImagen.shape[1]+ventana-1,self.dataImagen.shape[2]+ventana-1))
        paddingImage[:,math.floor((ventana-1)/2):self.dataImagen.shape[1]+math.floor((ventana-1)/2),math.floor((ventana-1)/2):self.dataImagen.shape[2]+math.floor((ventana-1)/2)] =  self.dataImagen #imagen aunmentada para padding
        
        for i in range(self.dataImagen.shape[1]):
            for j in range(self.dataImagen.shape[2]):
                if clase == self.groundTruth[i,j]:
                    datosClase.append(paddingImage[:,i:i+ventana,j:j+ventana].T) 
                    etiquetasClase.append(self.groundTruth[i,j])
        
        datosClase = np.array(datosClase)
        etiquetasClase = np.array(etiquetasClase)
        etiquetasClase = to_categorical(etiquetasClase, num_classes=numCls)
        return datosClase, etiquetasClase