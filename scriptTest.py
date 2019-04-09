from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from package.MorphologicalProfiles import morphologicalProfiles
from keras import layers
from keras import models
from keras import regularizers
import matplotlib.pyplot as plt

#CARGAR IMAGEN HSI Y GROUND TRUTH 
data = CargarHsi('PaviaU')
imagen = data.imagen 
groundTruth = data.groundTruth

#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis()
imagenPCA = pca.pca_calculate(imagen, varianza=0.95)
print(imagenPCA.shape)



####### EN DESARROLLO ##########################
# 1. Desarrollar una arquitectura de red convolucional en 3D
        # CNN 3D: hsiClassificationCNN3D.py  TEST_hsiClassificationCNN3D.py
        # Evaluar una red tipo Inception con CNN 3D
# 2. Desarrollar un esquema de data augmentation => prepararDatos.py
# 3. Evaluar el uso de SOM en el proceso de reduccion dimensional como etapa previa a la CNN.
# 4. Revisar documentacion reciente del estado del arte.