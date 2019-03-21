#Permite extraer conjuntos de entrenamiento, validacion y prueba en 1D, 2D y 3D
import math
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

class PrepararDatos:
    
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
        pass

    def extraerDatos1D(self,porEntrenamiento, porValidacion):
        #Datos de entrenamiento
        numCls = self.groundTruth.max()+1
        numtrain = self.indices.shape[1]*porEntrenamiento/100
        numtrain = math.floor(numtrain)
        datosEntrenamiento = np.zeros( (numtrain,self.dataImagen.shape[0]) )
        etiquetasEntrenamiento =  np.zeros( numtrain )
        numEntrenamiento = np.zeros((numCls,))
        for i in range(numtrain):
            datosEntrenamiento[i,:] = self.dataImagen[:,self.indices[0,i],self.indices[1,i]]
            etiquetasEntrenamiento[i] = self.groundTruth[self.indices[0,i],self.indices[1,i]]
            numEntrenamiento[self.groundTruth[self.indices[0,i],self.indices[1,i]]] += 1
        #Datos de validación
        numVal = self.indices.shape[1]*porValidacion/100
        numVal = math.floor(numVal)
        datosValidacion = np.zeros( (numVal,self.dataImagen.shape[0]) )
        etiquetasValidacion =  np.zeros( numVal )
        numValidacion = np.zeros((numCls,))
        for i in range(numVal):
            datosValidacion[i,:] = self.dataImagen[:,self.indices[0,numtrain+i],self.indices[1,numtrain+i]]
            etiquetasValidacion[i] = self.groundTruth[self.indices[0,numtrain+i],self.indices[1,numtrain+i]]
            numValidacion[self.groundTruth[self.indices[0,numtrain+i],self.indices[1,numtrain+i]]] += 1
        #Codificar etiquetas de entrenamiento y validacion ONE HOT
        etiquetasEntrenamiento = to_categorical(etiquetasEntrenamiento)
        etiquetasValidacion = to_categorical(etiquetasValidacion)    

        print('CANTIDAD DATOS ENTRENEMIENTO:'+ str(numEntrenamiento))
        print('CANTIDAD DATOS VALIDACIÓN: '+ str(numValidacion))

        return datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion
    
    def extraerDatos2D(self,porEntrenamiento, porValidacion, ventana):
        #Datos de entrenamiento
        numCls = self.groundTruth.max()+1
        numtrain = self.indices.shape[1]*porEntrenamiento/100
        numtrain = math.floor(numtrain)
        datosEntrenamiento = np.zeros( (numtrain,ventana,ventana,self.dataImagen.shape[0]) )
        etiquetasEntrenamiento =  np.zeros( numtrain )
        paddingImage = np.zeros((self.dataImagen.shape[0],self.dataImagen.shape[1]+ventana-1,self.dataImagen.shape[2]+ventana-1))
        paddingImage[:,math.floor((ventana-1)/2):self.dataImagen.shape[1]+math.floor((ventana-1)/2),math.floor((ventana-1)/2):self.dataImagen.shape[2]+math.floor((ventana-1)/2)] =  self.dataImagen #imagen aunmentada para padding
        numEntrenamiento = np.zeros((numCls,))
        for i in range(numtrain):
          datosEntrenamiento[i] = paddingImage[:,self.indices[0,i]:self.indices[0,i]+ventana,self.indices[1,i]:self.indices[1,i]+ventana].T
          etiquetasEntrenamiento[i] = self.groundTruth[self.indices[0,i],self.indices[1,i]]
          numEntrenamiento[self.groundTruth[self.indices[0,i],self.indices[1,i]]] += 1
        #Datos de validación
        numVal = self.indices.shape[1]*porValidacion/100
        numVal = math.floor(numVal)
        datosValidacion = np.zeros( (numVal,ventana,ventana,self.dataImagen.shape[0]) )
        etiquetasValidacion =  np.zeros( numVal )
        numValidacion = np.zeros((numCls,))
        for i in range(numVal):
            datosValidacion[i] = paddingImage[:,self.indices[0,numtrain+i]:self.indices[0,numtrain+i]+ventana,self.indices[1,numtrain+i]:self.indices[1,numtrain+i]+ventana].T
            etiquetasValidacion[i] = self.groundTruth[self.indices[0,numtrain+i],self.indices[1,numtrain+i]]
            numValidacion[self.groundTruth[self.indices[0,numtrain+i],self.indices[1,numtrain+i]]] += 1
        #Codificar etiquetas de entrenamiento y validacion ONE HOT
        etiquetasEntrenamiento = to_categorical(etiquetasEntrenamiento)
        etiquetasValidacion = to_categorical(etiquetasValidacion)        

        print('CANTIDAD DATOS ENTRENEMIENTO:'+ str(numEntrenamiento))
        print('CANTIDAD DATOS VALIDACIÓN: '+ str(numValidacion))

        return datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion

    def extraerDatos3D(self,porEntrenamiento, porValidacion, ventana):
        datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = self.extraerDatos2D(porEntrenamiento, porValidacion,ventana)
        datosEntrenamiento = datosEntrenamiento.reshape( (datosEntrenamiento.shape[0],datosEntrenamiento.shape[1],datosEntrenamiento.shape[2],datosEntrenamiento.shape[3],1) ) 
        datosValidacion = datosValidacion.reshape( (datosValidacion.shape[0],datosValidacion.shape[1],datosValidacion.shape[2],datosValidacion.shape[3],1) ) 
        return datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion

    def extraerDatosPrueba1D(self):
        indices = self.indices_org.copy()
        #Datos de Prueba
        numprueba = self.indices.shape[1]
        datosPrueba = np.zeros( (numprueba,self.dataImagen.shape[0]) )
        etiquetasPrueba =  np.zeros( numprueba )
        
        for i in range(numprueba):
            datosPrueba[i,:] = self.dataImagen[:,indices[0,i],indices[1,i]]
            etiquetasPrueba[i] = self.groundTruth[indices[0,i],indices[1,i]]

        return datosPrueba, etiquetasPrueba

    def extraerDatosPrueba2D(self,ventana):
        indices = self.indices_org.copy()
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

    def extraerDatosPrueba3D(self,ventana):
        datosPrueba, etiquetasPrueba = self.extraerDatosPrueba2D(ventana)
        datosPrueba = datosPrueba.reshape( (datosPrueba.shape[0],datosPrueba.shape[1],datosPrueba.shape[2],datosPrueba.shape[3],1) ) 
        return datosPrueba, etiquetasPrueba

    def extraerDatosClase1D(self, clase):
        datosClase = []
        etiquetasClase = []
        numCls = self.groundTruth.max()+1
        for i in range(self.dataImagen.shape[1]):
            for j in range(self.dataImagen.shape[2]):
                if clase == self.groundTruth[i,j]:
                    datosClase.append(self.dataImagen[:,i,j]) 
                    etiquetasClase.append(self.groundTruth[i,j])
        
        datosClase = np.array(datosClase)
        etiquetasClase = np.array(etiquetasClase)
        etiquetasClase = to_categorical(etiquetasClase, num_classes=numCls)
        return datosClase, etiquetasClase

    def extraerDatosClase2D(self, ventana, clase):
        datosClase = []
        etiquetasClase = []
        numCls = self.groundTruth.max()+1
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

    def extraerDatosClase3D(self, ventana, clase):
        datosClase, etiquetasClase = self.extraerDatosClase2D(ventana, clase)
        datosClase = datosClase.reshape( (datosClase.shape[0],datosClase.shape[1],datosClase.shape[2],datosClase.shape[3],1) ) 
        return datosClase, etiquetasClase

    def predictionToImage(self, dataPrediction):
        imgSalida = np.zeros((self.groundTruth.shape[0],self.groundTruth.shape[1]))
        datosSalida = dataPrediction.argmax(axis=1)
        
        for i in range(self.indices_org.shape[1]):
            imgSalida[self.indices_org[0,i],self.indices_org[1,i]] = datosSalida[i]
         
        return imgSalida
