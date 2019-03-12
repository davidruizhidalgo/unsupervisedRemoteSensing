#PRUEBA DE RED CONVOLUCIONAL 2D - CLASIFICACION HSI 
#Se utiliza PCA para reduccion dimensional y estraccion de caracteristicas espectrales. A la red convolucional se introduce
#una ventana sxs de la imagen original para la generacion de caracteristicas espaciales a partir de la convolucion. 
#Se utiliza como capa de salida un clasificador tipo Multinomial logistic regression. Todas las capas utilizan entrenamiento supervisado. 
from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from keras import layers 
from keras import models
import matplotlib.pyplot as plt
from keras.models import load_model

ventana = 9 #VENTANA 2D de PROCESAMIENTO
#CARGAR IMAGEN HSI Y GROUND TRUTH
data = CargarHsi('Indian_pines')
imagen = data.imagen
groundTruth = data.groundTruth

#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis(imagen)
imagenPCA = pca.pca_calculate(0.95) 

#PREPARAR DATOS PARA ENTRENAMIENTO
preparar = PrepararDatos(imagenPCA, groundTruth, False)
datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)
datosClase, etiquetasClase = preparar.extraerDatosClase2D(ventana,4) #Numero de la clase que se desea extraer

#CARGAR RED CONVOLUCIONAL
model = load_model('hsiClassificationCNN2D.h5')
print(model.summary())
#EVALUAR MODELO
test_loss, test_acc = model.evaluate(datosClase, etiquetasClase)
print(test_acc)

#GENERAR MAPA FINAL DE CLASIFICACIÓN
datosSalida = model.predict(datosPrueba)
datosSalida = preparar.predictionToImage(datosSalida)

#GRAFICAS
data.graficarHsi_VS(groundTruth, datosSalida)
