#PCA => Principal componet analysis using HSI
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class princiapalComponentAnalysis:
    
    def __init__(self,dataImagen):
        self.dataImagen = dataImagen.copy()

    def __str__(self):
        return f"Dimensiones Imagen:\t {self.dataImagen.shape}\n" 

    def pca_calculate(self,porcentaje):
        imageTemp = self.dataImagen.reshape((self.dataImagen.shape[0],self.dataImagen.shape[1]*self.dataImagen.shape[2])).T
        pca = PCA()
        pca.fit(imageTemp)
        imageTemp = pca.transform(imageTemp)
        
        #Evaluar el numero de coeficientes en base a los datos de varianza
        var = 0
        componentes = 0
        for i in range(pca.explained_variance_ratio_.shape[0]):
            var += pca.explained_variance_ratio_[i]
            if var > porcentaje:
                break
            else:
                componentes += 1
        
        imageTemp = imageTemp.reshape( (self.dataImagen.shape[1], self.dataImagen.shape[2],imageTemp.shape[1]) )
        imagePCA = np.zeros( (componentes, self.dataImagen.shape[1], self.dataImagen.shape[2]) )

        for i in range(imagePCA.shape[0]):
            imagePCA[i] = imageTemp[:,:,i]

        return imagePCA

    def graficarPCA(self,imagePCA, channel):
        plt.figure(1)
        plt.imshow(imagePCA[channel])
        plt.colorbar()
        plt.show()
    