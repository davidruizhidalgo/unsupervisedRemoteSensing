#Clase para cargar datos de una imagen HSI y retornarlos en formato numpy
#Permite graficar un canal de la imagen o el ground truth
#Permite normalizar los datos y codificarlos para usar en las redes 
#Permite extraer conjuntos de entrenamiento, validacion y prueba
import scipy.io as sio
import numpy as np

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
        print(data_t.shape)
        for i in range(data.shape[0]): # Transponer cada canal para ajustar los ejes coordenados
            data_t[i] = data[i].T 
        self.imagen = data_t
        #CARGAR GROUND TRUTH
        mat = sio.loadmat(dicData[name_data][2]) # Cargar archivo Ground Truth .mat
        data = np.array(mat[dicData[name_data][3]]) # Convertir Ground Truth a numpy array
        self.groundTruth = data

    def __str__(self):
        return f"Dimensiones:\t {self.imagen.shape}\n" \
               f"Datos:\t\t {self.imagen[0]}\n" \
               f"Dimensiones GT:\t {self.groundTruth.shape}\n" \
               f"Datos GT:\t\t {self.groundTruth}\n"

    def graficarHsi(self):
        print('FUNCION PARA GRAFICAR UN CANAL DE LA IMAGEN o el GT')