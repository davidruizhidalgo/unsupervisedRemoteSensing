#Script utilizado para el desarrollo de pruebas en el codigo. 
from package.cargarHsi import CargarHsi

#CARGAR IMAGEN HSI Y GROUND TRUTH
numTest = 10
dataSet = 'Salinas'
ventana = 9 #VENTANA 2D de PROCESAMIENTO
data = CargarHsi(dataSet)
imagen = data.imagen
groundTruth = data.groundTruth

data.graficarHsi_VS(groundTruth, imagen[15])

####### EN DESARROLLO ##########################
# 1. Desarrollar una arquitectura de red selección de caracteristicas NO SUPERVISADA
        # k-PCA
        # Autoendoder
        # Grafo de autoencoders
        # CNN entrenada de forma no supervisada
# 2. Desarrollar un esquema de data augmentation => prepararDatos.py
# 4. Revisar documentacion reciente del estado del arte.