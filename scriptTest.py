#Script utilizado para el desarrollo de pruebas en el codigo. 
#ENTRENAMIENTO DE RED CONVOLUCIONAL 2D - CLASIFICACION HSI 
#Se utiliza PCA para reduccion dimensional y estraccion de caracteristicas espectrales. A la red convolucional se introduce
#una ventana sxs de la imagen original para la generacion de caracteristicas espaciales a partir de la convolucion. 
#Se utiliza como capa de salida un clasificador tipo Multinomial logistic regression. Todas las capas utilizan entrenamiento supervisado. 
from package.cargarHsi import CargarHsi
from package.PCA import princiapalComponentAnalysis
import matplotlib.pyplot as plt
import numpy as np

#CARGAR IMAGEN HSI Y GROUND TRUTH
dataSet = 'Salinas'
ventana = 9 #VENTANA 2D de PROCESAMIENTO
data = CargarHsi(dataSet)
imagen = data.imagen
groundTruth = data.groundTruth

#ANALISIS DE COMPONENTES PRINCIPALES KPCA
pca = princiapalComponentAnalysis()
imagenKPCA = pca.kpca_calculate(imagen, componentes = 9) # 9,4,6
print(imagenKPCA.shape)
pca.graficarPCA(imagenKPCA,0)
####### EN DESARROLLO ##########################
# 1. Desarrollar una arquitectura de red selección de caracteristicas NO SUPERVISADA
        # k-PCA
        # Autoendoder
        # Grafo de autoencoders
        # CNN entrenada de forma no supervisada
# 2. Desarrollar un esquema de data augmentation => prepararDatos.py
# 4. Revisar documentacion reciente del estado del arte.