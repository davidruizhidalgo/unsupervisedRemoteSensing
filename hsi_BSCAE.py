#BRANCHES OF CONVOLUTIONAL STACKED AUTOENCODERS - INCEPTION BASED  
#Se implementa el encoder y el decoder convolucional utilizando la arquitectura de la red INCEPTION
# para aprender una representación de las caracteristicas de los datos de entrada.
#Con el encoder entrenado se implementa una capa de fine tunning para el ejuste de la ultima capa del clasificador. 
#El proceso utiliza una ventana sxs de la imagen original para la generacion de caracteristicas espaciales a partir de la convolucion. 
#Se utiliza como capa de salida un clasificador tipo Multinomial logistic regression.  
