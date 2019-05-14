#Permite cargar datos de una imagen HSI y retornarlos en formato numpy
#Permite normalizar los datos de entrada
#Permite graficar un canal de la imagen o el ground truth

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

class CargarHsi:
    
    def __init__(self,name_data):
        dicData = {'IndianPines' : ['C:/Users/david/Documents/dataSets/Indian_pines.mat', 'indian_pines_corrected', 'C:/Users/david/Documents/dataSets/Indian_pines_gt.mat', 'indian_pines_gt'],
                    'Salinas' : ['C:/Users/david/Documents/dataSets/Salinas.mat', 'salinas_corrected', 'C:/Users/david/Documents/dataSets/Salinas_gt.mat', 'salinas_gt'],
                    'Pavia' : ['C:/Users/david/Documents/dataSets/Pavia.mat', 'pavia', 'C:/Users/david/Documents/dataSets/Pavia_gt.mat', 'pavia_gt'],
                    'PaviaU' : ['C:/Users/david/Documents/dataSets/PaviaU.mat', 'paviaU', 'C:/Users/david/Documents/dataSets/PaviaU_gt.mat', 'paviaU_gt'], }
        #CARGAR CUBO DE DATOS
        mat = sio.loadmat(dicData[name_data][0]) # Cargar archivo .mat
        data = np.array(mat[dicData[name_data][1]]) # Convertir a numpy array
        data = data.T   # Transponer para ajustar los ejes coordenados      
        data_t = np.zeros( (data.shape[0],data.shape[2],data.shape[1]) )
        for i in range(data.shape[0]): # Transponer cada canal para ajustar los ejes coordenados
            data_t[i] = data[i].T 

        #NORMALIZAR DATOS DE ENTRADA
        mean = data_t.mean(axis=0)
        data_t -= mean
        std = data_t.std(axis=0)
        data_t /= std
        self.imagen = data_t.copy()   #IMAGEN DE ENTRADA NORMALIZADA
        #CARGAR GROUND TRUTH
        mat = sio.loadmat(dicData[name_data][2]) # Cargar archivo Ground Truth .mat
        data = np.array(mat[dicData[name_data][3]]) # Convertir Ground Truth a numpy array
        self.groundTruth = data.copy()         #GROUND TRUTH

    def __str__(self):
        pass

    def graficarHsi(self,imageChannel):
        plt.figure()
        plt.imshow(imageChannel)
        plt.colorbar()
        plt.show()

    def graficarHsi_VS(self, img_1, img_2, cmap ='fc'):
        plt.figure(1)
        plt.subplot(1,2,1)
        plt.imshow(img_1)
        plt.subplot(1,2,2)
        if cmap == 'fc':
            plt.imshow(img_2)
        else:
            plt.imshow(img_2, cmap='Greys',  interpolation='nearest')
        plt.show()