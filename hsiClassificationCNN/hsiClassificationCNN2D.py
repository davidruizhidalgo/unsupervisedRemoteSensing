#RED CONVOLUCIONAL 2D - CLASIFICACION HSI 
from paquete.cargarHsi import CargarHsi
from paquete.prepararDatos import PrepararDatos
from paquete.PCA import princiapalComponentAnalysis

#CARGAR IMAGEN HSI Y GROUND TRUTH
data = CargarHsi('Salinas')
imagen = data.imagen
groundTruth = data.groundTruth

#PREPARAR DATOS PARA ENTRENAMIENTO
preparar = PrepararDatos(imagen, groundTruth)
datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,30)

#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis(imagen)
imagenPCA = pca.pca_calculate(6)
pca.graficarPCA(imagenPCA,0)


#GRAFICAR GROUND TRUTH
data.graficarHsi(groundTruth)