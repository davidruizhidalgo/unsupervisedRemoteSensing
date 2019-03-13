#ARCHIVO DE PRUEBA PARA LAS FUNCIONES DESARROLLADAS EN EL PAQUETE
from package.cargarHsi import CargarHsi
from package.PCA import princiapalComponentAnalysis
import matplotlib.pyplot as plt 

ventana = 9 #VENTANA 2D de PROCESAMIENTO
clases = 17 #CLASES PRESENTES EN LA IMAGEN

#CARGAR IMAGEN HSI Y GROUND TRUTH 
data = CargarHsi('Indian_pines')
imagen = data.imagen 
groundTruth = data.groundTruth

#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis(imagen)
imagenPCA = pca.pca_calculate(0.95)


####### EN DESARROLLO ##########################
# 2. Desarrollar un esquema de data augmentation => prepararDatos.py
# 3. Implementar extracion de caracterristicas espaciales con Atributte Profiles.
# 4. Evaluar el uso de SOM en el proceso de reduccion dimensional como etapa previa a la CNN.
# 5. Revisar documentacion reciente del estado del arte.