#PCA => Principal componet analysis using HSI
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA

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

    def kpca_calculate(self, imagen_in, componentes):
        #TOMA LA PORCION DE LA IMAGEN DE TAMAÑO W
        i = 0 #Indice x para la imagen
        j = 0 #Indice y para la imagen  
        W = 50 #Tamaño de subconjunto 50  por indices
        n_componentes = 0 #Numero inicial de componentes principales

        for i in range(imagen_in.shape[1]): #Recorrer x
            i_l = i*W
            i_h = (i+1)*W
            if i_l >= imagen_in.shape[1]:
                break
            if i_h > imagen_in.shape[1]:
                i_h = imagen_in.shape[1]
            for j in range(imagen_in.shape[2]):  #Recorrer y
                j_l = j*W
                j_h = (j+1)*W
                if j_l >= imagen_in.shape[2]:
                    break
                if j_h > imagen_in.shape[2]:
                    j_h = imagen_in.shape[2]
                
                dataImagen = imagen_in[:, i_l:i_h, j_l:j_h]
                imageTemp = dataImagen.reshape((dataImagen.shape[0],dataImagen.shape[1]*dataImagen.shape[2])).T  #Reorganiza para aplicar KPCA
                #APLICA KPCA SOBRE TODOS LOS ELEMENTOS DIMENSIONALES 
                kpca = KernelPCA( kernel='rbf' ) # n_components=None, gamma=0.01
                X_transformed = kpca.fit_transform(imageTemp)
                #Calcula el porcentaje de varianza de cada componente y el número de componentes a utilizar
                if n_componentes == 0:
                    sum_varianza = 0
                    varianza  = kpca.lambdas_/np.sum(kpca.lambdas_)
                    for v in range(varianza.shape[0]):
                        sum_varianza = sum_varianza+varianza[v]
                        if sum_varianza > 0.95:
                            break
                        else:
                            n_componentes += 1
                    if n_componentes < componentes:
                       n_componentes = componentes
                    if n_componentes >  imagen_in.shape[1]/2:
                       n_componentes = componentes  
                    ImagenOut = np.zeros( (n_componentes, imagen_in.shape[1], imagen_in.shape[2]) )    
                #RECUPERA EL NUMERO DE COMPONENTES NECESARIO
                imageTemp = X_transformed[:,0:n_componentes].reshape( (dataImagen.shape[1], dataImagen.shape[2],n_componentes) )
                imageKPCA = np.zeros( (n_componentes, dataImagen.shape[1], dataImagen.shape[2]) )
                # RECONTRUIR LA SALIDA EN LA FORMA DE LA IMAGEN DE ENTRADA
                for i in range(imageKPCA.shape[0]):
                    imageKPCA[i] = imageTemp[:,:,i]
                ImagenOut[:, i_l:i_h, j_l:j_h] = imageKPCA
        return ImagenOut
        
    def graficarPCA(self,imagePCA, channel):
        plt.figure(1)
        plt.imshow(imagePCA[channel])
        plt.colorbar()
        plt.show()
    