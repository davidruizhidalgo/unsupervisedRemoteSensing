#ANALISIS DE FIRMAS ESPECTRALES DE CADA UNA DE LAS CLASES EN LA HSI
from package.cargarHsi import CargarHsi
from package.firmasEspectrales import FirmasEspectrales
import matplotlib.pyplot as plt 

from package.PCA import princiapalComponentAnalysis 

ventana = 9 #VENTANA 2D de PROCESAMIENTO
clases = 17 #CLASES PRESENTES EN LA IMAGEN
#CARGAR IMAGEN HSI Y GROUND TRUTH 
data = CargarHsi('Indian_pines')
imagen = data.imagen 
groundTruth = data.groundTruth

#ANALISIS DE COMPONENTES PRINCIPALES 
pca = princiapalComponentAnalysis(imagen)
imagenPCA = pca.pca_calculate(0.95)

#PREPARAR DATOS PARA ENTRENAMIENTO
preparar = FirmasEspectrales(imagen, groundTruth, False)
print(preparar) 