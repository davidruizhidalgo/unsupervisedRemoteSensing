#Permite cargar datos de una imagen HSI y retornarlos en formato numpy
#Permite normalizar los datos de entrada
#Permite graficar un canal de la imagen o el ground truth
#CARGA  EL CUBO DE DATOS HSI Y EL GROUND TRUTH
#PERMITE GRAFICAR UN CANAL DE LA IMAGEN O DOS IMAGENES EN UNA MISMA VENTANA
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

class CargarHsi:
    
    def __init__(self,name_data):
        dicData = {'Indian_pines' : ['C:/Users/david/Documents/SoftwareDesarrollado/dataSets/indian_pines.mat', 'indian_pines_corrected', 'C:/Users/david/Documents/SoftwareDesarrollado/dataSets/indian_pines_gt.mat', 'indian_pines_gt'],
                    'Salinas' : ['C:/Users/david/Documents/SoftwareDesarrollado/dataSets/salinas.mat', 'salinas_corrected', 'C:/Users/david/Documents/SoftwareDesarrollado/dataSets/salinas_gt.mat', 'salinas_gt'],
                    'Pavia' : ['C:/Users/david/Documents/SoftwareDesarrollado/dataSets/pavia.mat', 'pavia', 'C:/Users/david/Documents/SoftwareDesarrollado/dataSets/pavia_gt.mat', 'pavia_gt'], }
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
        return f"Dimensiones Imagen:\t {self.imagen.shape}\n" \
               f"Datos:\t\t {self.imagen[0]}\n" \
               f"Dimensiones Ground Truth:\t {self.groundTruth.shape}\n" \
               f"Datos GT:\t\t {self.groundTruth}\n"

    def graficarHsi(self,imageChannel):
        plt.figure()
        plt.imshow(imageChannel)
        plt.show()

    def graficarHsi_VS(self, img_1, img_2):
        plt.figure(1)
        plt.subplot(1,2,1)
        plt.imshow(img_1)
        plt.subplot(1,2,2)
        plt.imshow(img_2)
        plt.show()