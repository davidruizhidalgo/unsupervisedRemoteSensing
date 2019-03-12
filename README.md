# Sistema no Supervisado para Sensado Remoto Usando HSI
Sensado remoto de vegetación y cultivos usando imágenes hiperespectrales y métodos de entrenamiento no supervisados

Este proyecto contiene diferentes aplicaciones con código desarrollado en python.

## Pasos de Instalación para el funcionamiento del Software Desarrollado
1. Descargar e instalar Anaconda.exe
2. conda create -n NOMBRE_DEL_ESPACIO_DE_TRABAJO
3. activate NOMBRE_DEL_ESPACIO_DE_TRABAJO
4. conda install tensorflow-gpu
5. conda install keras
6. conda install matplotlib
7. conda install pylint
8. conda install scikit-learn

### Sincronización con Repositorio en GitHub
Descargar git.exe de https://git-scm.com/ y en la carpeta contenedora del proyecto introducir los siguientes comandos:
1. git clone https://github.com/davidruizhidalgo/unsupervisedRemoteSensing.git
2. git add.
3. git commit -m "Mensaje especificando características de la actualización"
        Si es la primera vez en el repositorio se deben introducir los datos de usuario e email con los comandos 
        sugeridos en la línea de comandos. 
4. git push origin master
### Notas: 
- La programación de los scripts es realizada en visual studio code
- Tener en cuenta el directorio donde se encuentran los datasets. En este caso: C:/Users/david/Documents/dataSets/
## 1. Package
Paquete que contiene diferentes funciones utilizadas para el procesamiento de las imágenes hiperespectrales:

cargarHSI.py => Permite cargar un archivo .mat con la imagen hiperespectral y el groundtruth. Realiza la normalización de los valores de entrada utilizando la media y la desviación estándar de cada firma espectral. Esto permite obtener datos con media cero y desviación uno. El archivo implementa también funciones de graficar una o dos imágenes de un solo canal.

prepararDatos.py => Permite extraer conjuntos de datos de entrenamiento, validación y prueba en 1D y 2D, utilizando ventanas y porcentajes de datos variables.

PCA.py => Implementa el análisis de componentes principales sobre la imagen hiperespectral de entrada.

firmasEspectrales.py => Recoge las firmas espectrales de cada una de las clases en la imagen. Calcula el promedio y grafica las firmas espectrales con el objetivo de determinar posibles diferencias en las curvas. 

## 2. Análisis de firmas espectrales 
analisisEspectral.py => Permite realizar el análisis de las firmas espectrales de cada una de las clases presentes en la imagen hiperespectral. Se promedia la firma espectral de cada clase y se grafican para observar las diferencias de los espectros.
## 3. Clasificación de una HSI usando PCA y una 2D CNN
hsiClassificationCNN2D.py => Entrenamiento de una  red convolucional 2d para clasificación usando HSI. Se utiliza PCA para reduccion dimensional y estraccion de caracteristicas espectrales. A la red convolucional se introduce una ventana sxs de la imagen original para la generacion de caracteristicas espaciales a partir de la convolucion. Se utiliza como capa de salida un clasificador tipo Multinomial logistic regression. Todas las capas utilizan entrenamiento supervisado. 

TEST_hsiClassificationCNN2D.py => Carga y ejecución de una  red convolucional 2d para clasificación usando HSI. 

## 4. Clasificación de una HSI usando un Modelo Inseption
.....