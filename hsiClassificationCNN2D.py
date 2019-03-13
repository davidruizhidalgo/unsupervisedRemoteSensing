#ENTRENAMIENTO DE RED CONVOLUCIONAL 2D - CLASIFICACION HSI 
#Se utiliza PCA para reduccion dimensional y estraccion de caracteristicas espectrales. A la red convolucional se introduce
#una ventana sxs de la imagen original para la generacion de caracteristicas espaciales a partir de la convolucion. 
#Se utiliza como capa de salida un clasificador tipo Multinomial logistic regression. Todas las capas utilizan entrenamiento supervisado. 
from package.cargarHsi import CargarHsi
from package.prepararDatos import PrepararDatos
from package.PCA import princiapalComponentAnalysis
from keras import layers
from keras import models
from keras import regularizers
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
preparar = PrepararDatos(imagenPCA, groundTruth, False)
datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,30,ventana)
datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(ventana)

#DEFINICION RED CONVOLUCIONAL
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), kernel_regularizer=regularizers.l2(0.001),activation='relu', input_shape=(datosEntrenamiento.shape[1],datosEntrenamiento.shape[2],datosEntrenamiento.shape[3])))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.001),activation='relu'))
#model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(64, (3, 3), kernel_regularizer=regularizers.l2(0.001),activation='relu'))
#AÑADE UN CLASIFICADOR MLR EN EL TOPE DE LA CONVNET
model.add(layers.Flatten())
model.add(layers.Dense(64, kernel_regularizer=regularizers.l2(0.001), activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(groundTruth.max()+1, activation='softmax'))
print(model.summary())

#ENTRENAMIENTO DE LA RED CONVOLUCIONAL
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(datosEntrenamiento,etiquetasEntrenamiento,epochs=15,batch_size=512,validation_data=(datosValidacion, etiquetasValidacion))

#EVALUAR MODELO
test_loss, test_acc = model.evaluate(datosPrueba, etiquetasPrueba)
print(test_acc)
#GENERAR MAPA FINAL DE CLASIFICACIÓN
datosSalida = model.predict(datosPrueba)
datosSalida = preparar.predictionToImage(datosSalida)

#GRAFICAR TRAINING AND VALIDATION LOSS
plt.figure(1)
plt.subplot(211)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(212)
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#GRAFICAS
data.graficarHsi_VS(groundTruth, datosSalida)

#GUARDAR MODELO DE RED CONVOLUCIONAL
model.save('hsiClassificationCNN2D.h5')
