#RED CONVOLUCIONAL 2D - CLASIFICACION HSI 
from paquete.cargarHsi import CargarHsi
from paquete.prepararDatos import PrepararDatos
from paquete.PCA import pcaAnalysis

#CARGAR IMAGEN HSI Y GROUND TRUTH
data = CargarHsi('Indian_pines')
imagen = data.imagen
groundTruth = data.groundTruth

#PREPARAR DATOS PARA ENTRENAMIENTO
preparar = PrepararDatos(imagen, groundTruth)
datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,30)

#ANALISIS DE COMPONENTES PRINCIPALES
pca = pcaAnalysis(imagen)
print(pca)

#GRAFICAR GROUND TRUTH
data.graficarHsi(groundTruth)