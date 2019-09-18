#Script utilizado para el desarrollo de pruebas en el codigo. 
####### EN DESARROLLO ##########################
# 1. Desarrollar una arquitectura de red selección de caracteristicas NO SUPERVISADA
        # CNN Autoendoder
        # Grafo de CNN autoencoders
        # Evaluar funciones de distancia en la funcion de costo
        # Evaluar estrategias de concatenacion en BSCAE
        # Evaluar el uso de Extinction Profiles como una mejora de EAP => Corregir EAP
# 2. Desarrollar un esquema de data augmentation => prepararDatos.py
# 4. Revisar documentacion reciente del estado del arte.

from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from package.MorphologicalProfiles import morphologicalProfiles

import matplotlib.pyplot as plt
import numpy as np

import siamxt  

#CARGAR IMAGEN HSI Y GROUND TRUTH
dataSet = 'IndianPines'
data = CargarHsi(dataSet)
imagen = data.imagen
groundTruth = data.groundTruth

#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis()
#imagenPCA = pca.pca_calculate(imagen, varianza=0.9)
imagenPCA = pca.pca_calculate(imagen, componentes=1)

imagenPCA = (imagenPCA-imagenPCA.min())/(imagenPCA.max()-imagenPCA.min()) *255
imagen = imagenPCA.astype(np.uint8)


#Structuring element with connectivity-8
Bc = np.ones((3,3),dtype = bool)
#Building the max-tree with the connectivity defined
mxt = siamxt.MaxTreeAlpha(imagen[0],Bc)
print(mxt.shape)
print("Number of max-tree nodes: %d" %mxt.node_array.shape[1])
print("Number of max-tree leaves: %d" %(mxt.node_array[1,:] == 0).sum())

'''
import matlab.engine 
eng = matlab.engine.start_matlab() 
tf = eng.isprime(37) 
print(tf) 
'''

'''
from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from package.MorphologicalProfiles import morphologicalProfiles

import matplotlib.pyplot as plt
import numpy as np


#CARGAR IMAGEN HSI Y GROUND TRUTH
dataSet = 'IndianPines'
data = CargarHsi(dataSet)
imagen = data.imagen
groundTruth = data.groundTruth

#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis()
#imagenPCA = pca.pca_calculate(imagen, varianza=0.9)
imagenPCA = pca.pca_calculate(imagen, componentes=1)
print(imagenPCA.shape)

#ESTIMACIÓN DE EXTENDED ATTRIBUTE PROFILES
mp = morphologicalProfiles()
imagenEAP = mp.EAP(imagenPCA, num_thresholds=10)  
print(imagenEAP.shape)

data.graficarHsi_VS(imagenEAP[3], imagenEAP[14])
'''