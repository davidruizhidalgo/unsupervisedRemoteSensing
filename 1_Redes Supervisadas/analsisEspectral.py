#ANALISIS DE FIRMAS ESPECTRALES DE CADA UNA DE LAS CLASES EN LA HSI
from package.cargarHsi import CargarHsi
from package.firmasEspectrales import FirmasEspectrales
from package.MorphologicalProfiles import morphologicalProfiles
from package.PCA import princiapalComponentAnalysis
import matplotlib.pyplot as plt 

ventana = 9 #VENTANA 2D de PROCESAMIENTO
#CARGAR IMAGEN HSI Y GROUND TRUTH 
data = CargarHsi('Indian_pine')
imagen = data.imagen 
groundTruth = data.groundTruth

#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis()
imagenPCA = pca.pca_calculate(imagen, varianza=0.95)

#ESTIMACIÓN DE EXTENDED ATTRIBUTE PROFILES
mp = morphologicalProfiles()
imagenEAP = mp.EAP(imagenPCA)

#PROMEDIAR FIRMAS ESPECTRALES
espectros = FirmasEspectrales(imagen, groundTruth)
firmas = espectros.promediarFirmas() # Promedios de todas las firmas espectrales
#firmasclase = espectros.firmasClase(0) # Arreglo con las firmas de la clase n

#GRAFICAR FIRMAS ESPECTRALES
espectros.graficarFirmas(firmas) # Grafica de los promedios de todas las firmas espectrales
#espectros.graficarFirmas(firmasclase) #Grafica del arreglo con las firmas de la clase n
