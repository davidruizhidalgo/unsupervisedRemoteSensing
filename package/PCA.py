#PCA => Principal componet analysis using HSI
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class princiapalComponentAnalysis:
    
    def __init__(self):
        pass

    def __str__(self):
        pass

    def pca_calculate(self,imagen_in,varianza = None,componentes = None):
        dataImagen = imagen_in.copy()
        if varianza != None :
            imageTemp = dataImagen.reshape((dataImagen.shape[0],dataImagen.shape[1]*dataImagen.shape[2])).T
            pca = PCA()
            pca.fit(imageTemp)
            imageTemp = pca.transform(imageTemp)
            #Evaluar el numero de coeficientes en base a los datos de varianza
            var = 0
            num_componentes = 0
            for i in range(pca.explained_variance_ratio_.shape[0]):
                var += pca.explained_variance_ratio_[i]
                if var > varianza:
                    break
                else:
                    num_componentes += 1
            imageTemp = imageTemp.reshape( (dataImagen.shape[1], dataImagen.shape[2],dataImagen.shape[0]) )
            imagePCA = np.zeros( (num_componentes, dataImagen.shape[1], dataImagen.shape[2]) )
            for i in range(imagePCA.shape[0]):
                imagePCA[i] = imageTemp[:,:,i]
        if componentes != None:
            imageTemp = dataImagen.reshape((dataImagen.shape[0],dataImagen.shape[1]*dataImagen.shape[2])).T
            c_pca = PCA(n_components=componentes)
            c_pca.fit(imageTemp)
            imageTemp = c_pca.transform(imageTemp)
            imageTemp = imageTemp.reshape( (dataImagen.shape[1], dataImagen.shape[2],imageTemp.shape[1]) )
            imagePCA = np.zeros( (componentes, dataImagen.shape[1], dataImagen.shape[2]) )
            for i in range(imagePCA.shape[0]):
                imagePCA[i] = imageTemp[:,:,i]
        return imagePCA

    def graficarPCA(self,imagePCA, channel):
        plt.figure(1)
        plt.imshow(imagePCA[channel])
        plt.colorbar()
        plt.show()
    