#RED CONVOLUCIONAL 2D - CLASIFICACION HSI 
from paquete.cargarHsi import CargarHsi

#CARGAR IMAGEN HSI Y GROUND TRUTH
data = CargarHsi('Indian_pines')

imagen = data.imagen
groundTruth = data.groundTruth

#GRAFICAR GROUND TRUTH
data.graficarHsi(groundTruth)