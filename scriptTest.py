#ARCHIVO DE PRUEBA PARA LAS FUNCIONES DESARROLLADAS EN EL PAQUETE
from package.cargarHsi import CargarHsi
from package.PCA import princiapalComponentAnalysis
from package.MorphologicalProfiles import morphologicalProfiles
import matplotlib.pyplot as plt 

ventana = 9 #VENTANA 2D de PROCESAMIENTO
clases = 17 #CLASES PRESENTES EN LA IMAGEN

#CARGAR IMAGEN HSI Y GROUND TRUTH 
data = CargarHsi('Indian_pines')
imagen = data.imagen 
groundTruth = data.groundTruth

#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis()
imagenPCA = pca.pca_calculate(imagen, varianza=0.95)

#ESTIMACIÓN DE EXTENDED ATTRIBUTE PROFILES
mp = morphologicalProfiles()
imagenEAP = mp.EAP(imagenPCA)

####### EN DESARROLLO ##########################
# 1. Desarrollar una arquitectura de red convolucional en 3D
        # CNN 3D: hsiClassificationCNN3D.py  TEST_hsiClassificationCNN3D.py
        # Evaluar una red tipo Inception con CNN 3D
# 2. Desarrollar un esquema de data augmentation => prepararDatos.py
# 3. Evaluar el uso de SOM en el proceso de reduccion dimensional como etapa previa a la CNN.
# 4. Revisar documentacion reciente del estado del arte.