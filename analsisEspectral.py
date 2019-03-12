#ANALISIS DE FIRMAS ESPECTRALES DE CADA UNA DE LAS CLASES EN LA HSI
from package.cargarHsi import CargarHsi
from package.firmasEspectrales import FirmasEspectrales
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

#PROMEDIAR FIRMAS ESPECTRALES
espectros = FirmasEspectrales(imagen, groundTruth, clases)
firmas = espectros.promediarFirmas() # Promedios de todas las firmas espectrales
#firmasclase = espectros.firmasClase(1) # Arreglo con las firmas de la clase n

#GRAFICAR FIRMAS ESPECTRALES
espectros.graficarFirmas(firmas) # Grafica de los promedios de todas las firmas espectrales
#espectros.graficarFirmas(firmasclase) #Grafica del arreglo con las firmas de la clase n
