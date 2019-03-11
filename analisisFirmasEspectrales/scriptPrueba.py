#ARCHIVO DE PRUEBA PARA FUNCIONES EN LAS CLASES
from paquete.cargarHsi import CargarHsi
from paquete.prepararDatos import PrepararDatos
from paquete.PCA import princiapalComponentAnalysis
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

#PREPARAR DATOS PARA ENTRENAMIENTO
preparar = PrepararDatos(imagenPCA, groundTruth, False)


######### REVISAR ######################
# Datos de entrenamiento aleatorios en CADA clase => prepararDatos.py
# Regularizar con data augmentation               => prepararDatos.py
# Clase para analsis de firmas espectrales        => analisisEspectral.py firmasEspectrales.py
# Continuar con otras redes profundas y arquitecturas