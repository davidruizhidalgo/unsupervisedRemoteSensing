#Permite cargar datos de una imagen HSI y retornarlos en formato numpy
#Permite normalizar los datos de entrada
#Permite graficar un canal de la imagen o el ground truth

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

class CargarHsi:
    
    def __init__(self,name_data):
        dicData = {'IndianPines' : ["../dataSets/Indian_pines.mat", 'indian_pines_corrected', "../dataSets/Indian_pines_gt.mat", 'indian_pines_gt'],
                    'Salinas' : ["../dataSets/Salinas.mat", 'salinas_corrected', "../dataSets/Salinas_gt.mat", 'salinas_gt'],
                    'SalinasA' : ["../dataSets/SalinasA.mat", 'salinasA_corrected', "../dataSets/SalinasA_gt.mat", 'salinasA_gt'],
                    'Pavia' : ["../dataSets/Pavia.mat", 'pavia', "../dataSets/Pavia_gt.mat", 'pavia_gt'],
                    'PaviaU' : ["../dataSets/PaviaU.mat", 'paviaU', "../dataSets/PaviaU_gt.mat", 'paviaU_gt'], 
                    'Urban210' : ["../dataSets/Urban210.mat", 'imagenOut', "../dataSets/Urban210_gt.mat", 'imagenOut_gt'],
                    'Samson' : ["../dataSets/samson.mat", 'imagenOut', "../dataSets/samson_gt.mat", 'imagenOut_gt'],
                    'Jasper' : ["../dataSets/Jasper.mat", 'imagenOut', "../dataSets/Jasper_gt.mat", 'imagenOut_gt'],}
        #CARGAR CUBO DE DATOS
        mat = sio.loadmat(dicData[name_data][0]) # Cargar archivo .mat
        data = np.array(mat[dicData[name_data][1]]) # Convertir a numpy array
        data = data.T   # Transponer para ajustar los ejes coordenados      
        data_t = np.zeros( (data.shape[0],data.shape[2],data.shape[1]) )
        for i in range(data.shape[0]): # Transponer cada canal para ajustar los ejes coordenados
            data_t[i] = data[i].T 

        #NORMALIZAR DATOS DE ENTRADA
        #data_t = (data_t-data_t.min())/(data_t.max()-data_t.min())
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

    def graficar_history(self, history):
        if history != 0:
            acc = history.history['acc']
            val_acc = history.history['val_acc']
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            plt.figure(1)
            plt.subplot(211)
            epochs = range(1, len(loss) + 1)
            plt.plot(epochs, loss, 'bo', label='Training loss')
            plt.plot(epochs, val_loss, 'b', label='Validation loss')
            plt.title('Training and validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(212)
            plt.plot(epochs, acc, 'bo', label='Training acc')
            plt.plot(epochs, val_acc, 'b', label='Validation acc')
            plt.title('Training and validation accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()