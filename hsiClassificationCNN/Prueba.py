#ARCHIVO DE PRUEBA PARA CLASES EN LOS PAQUETES
from paquete.cargarHsi import CargarHsi
from paquete.prepararDatos import PrepararDatos
from paquete.prepararDatosSinFondo import PrepararDatosSinFondo
from paquete.PCA import princiapalComponentAnalysis
import matplotlib.pyplot as plt

ventana = 9 #VENTANA 2D de PROCESAMIENTO
#CARGAR IMAGEN HSI Y GROUND TRUTH
data = CargarHsi('Indian_pines')
imagen = data.imagen
groundTruth = data.groundTruth

#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis(imagen)
imagenPCA = pca.pca_calculate(0.95)

#PREPARAR DATOS PARA ENTRENAMIENTO
preparar = PrepararDatos(imagenPCA, groundTruth)
datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos1D(50,30)

print(etiquetasEntrenamiento.shape)
print(etiquetasValidacion.shape)
