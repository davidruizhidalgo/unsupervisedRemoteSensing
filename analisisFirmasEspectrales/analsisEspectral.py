#ANALISIS DE FIRMAS ESPECTRALES DE CADA UNA DE LAS CLASES EN LA HSI
from paquete.cargarHsi import CargarHsi
from paquete.firmasEspectrales import FirmasEspectrales
import matplotlib.pyplot as plt

ventana = 9 #VENTANA 2D de PROCESAMIENTO
clases = 17 #CLASES PRESENTES EN LA IMAGEN
#CARGAR IMAGEN HSI Y GROUND TRUTH
data = CargarHsi('Indian_pines')
imagen = data.imagen
groundTruth = data.groundTruth

#PREPARAR DATOS PARA ENTRENAMIENTO
preparar = FirmasEspectrales(imagen, groundTruth, False)