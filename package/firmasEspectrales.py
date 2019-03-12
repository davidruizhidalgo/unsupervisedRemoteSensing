#Permite extraer conjuntos de entrenamiento, validacion y prueba en 1D y 2D
#Trabaja con la informacion del fondo (clase cero) o la descarta para el procesamiento de los datos.
import math
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical

class FirmasEspectrales:
    
    def __init__(self, dataImagen, groundTruth):
        self.dataImagen = dataImagen
        self.groundTruth = groundTruth
        self.numclases = groundTruth.max()+1
               
    def __str__(self):
        return f"Dimensiones Imagen:\t {self.dataImagen.shape}\n" \
               f"Dimensiones Ground Truth:\t {self.groundTruth.shape}\n" \
               f"Número de Clases:\t {self.numclases}\n"

    def promediarFirmas(self):
        firmas = np.zeros((self.numclases,self.dataImagen.shape[0]))
        numElementos = np.zeros((self.numclases))
        
        for i in range(self.dataImagen.shape[1]):
            for j in range(self.dataImagen.shape[2]):
                firmas[self.groundTruth[i,j]] += self.dataImagen[:,i,j]
                numElementos[self.groundTruth[i,j]] += 1
        
        for k in range (self.numclases):
            firmas[k] /= numElementos[k]
    
        return firmas

    def firmasClase(self, clase):
        classSignature = []
        for i in range(self.dataImagen.shape[1]):
            for j in range(self.dataImagen.shape[2]):
                if self.groundTruth[i,j] == clase:
                    classSignature.append(self.dataImagen[:,i,j]) 
        classSignature = np.array(classSignature)

        return classSignature

    def graficarFirmas(self, firmas):
        bandas = np.linspace(1, firmas.shape[1], firmas.shape[1])

        if firmas.shape[0] > self.numclases:
            for i in range (firmas.shape[0]):
                plt.plot(bandas, firmas[i])
                plt.title("Firmas Espectrales de la Clase")
        else:
            for i in range (firmas.shape[0]):
                plt.plot(bandas, firmas[i], label='Clase '+str(i))
                plt.title("Firmas Espectrales por Clase")
                plt.legend()
        
        plt.xlabel('Banda Espectral')
        plt.ylabel('Valor')
 
        plt.grid()
        plt.show()