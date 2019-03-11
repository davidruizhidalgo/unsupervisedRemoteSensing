#RED PROFUNDA CON TOPOLOGIA DE GRAFO INCEPTION - CLASIFICACION HSI 
#Se utiliza la topologia de red de grafos INCEPTION para la inclusion de caracteristicas espectrales y espaciales en la arquitectura de 
#la red profunda. Esto se logra utilizando redes convolucionales con diferentes tamaños de ventana; 1x1 para el manejo de caracteristicas 
#espectrales y 3x3 o 2x2 para el manejo de posibles dependencias espaciales. Se extrae entonces un tensor 4D utilizando una ventana sxs 
#de la imagen original.
#Se utiliza como capa de salida un clasificador tipo Multinomial logistic regression. Todas las capas utilizan entrenamiento supervisado. 
from paquete.cargarHsi import CargarHsi
from paquete.prepararDatos import PrepararDatos
from paquete.PCA import princiapalComponentAnalysis
from keras import layers
from keras.models import Model
from keras.layers import Input
import matplotlib.pyplot as plt

#CARGAR IMAGEN HSI Y GROUND TRUTH
data = CargarHsi('Indian_pines')
imagen = data.imagen
groundTruth = data.groundTruth

#ANALISIS DE COMPONENTES PRINCIPALES
pca = princiapalComponentAnalysis(imagen)
imagenPCA = pca.pca_calculate(0.95)

#PREPARAR DATOS PARA ENTRENAMIENTO
preparar = PrepararDatos(imagenPCA, groundTruth)
datosEntrenamiento, etiquetasEntrenamiento, datosValidacion, etiquetasValidacion = preparar.extraerDatos2D(50,30,9)
datosPrueba, etiquetasPrueba = preparar.extraerDatosPrueba2D(9)

#DEFINICION RED CONVOLUCIONAL
#input_tensor = Input(shape=(datosEntrenamiento.shape[1],datosEntrenamiento.shape[2],datosEntrenamiento.shape[3]))
input_tensor = Input(shape=(28,28,3))
# We assume the existence of a 4D input tensor 
# Every branch has the same stride value (2), which is necessary to keep all
# branch outputs the same size, so as to be able to concatenate them.
branch_a = layers.Conv2D(128, 1, activation='relu', strides=2)(input_tensor)
# In this branch, the striding occurs in the spatial convolution layer
branch_b = layers.Conv2D(128, 1, activation='relu')(input_tensor)
branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b)
# In this branch, the striding occurs in the average pooling layer
branch_c = layers.AveragePooling2D(3, strides=2)(input_tensor)
branch_c = layers.Conv2D(128, 3, activation='relu')(branch_c)
branch_d = layers.Conv2D(128, 1, activation='relu')(input_tensor)
branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d)
branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)
# Finally, we concatenate the branch outputs to obtain the module output
output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)
#model = Model(input_tensor,output)
#print(model.summary())
'''
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
model.add(layers.Dropout(0.5))
model.add(layers.Dense(17, activation='softmax'))
print(model.summary())
'''
'''
#ENTRENAMIENTO DE LA RED CONVOLUCIONAL
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(datosEntrenamiento,etiquetasEntrenamiento,epochs=15,batch_size=512,validation_data=(datosValidacion, etiquetasValidacion))

#EVALUAR MODELO
test_loss, test_acc = model.evaluate(datosPrueba, etiquetasPrueba)
print(test_acc)
#GENERAR MAPA FINAL DE CLASIFICACIÓN
datosSalida = model.predict(datosPrueba)
datosSalida = datosSalida.argmax(axis=1)
datosSalida = datosSalida.reshape((groundTruth.shape[0],groundTruth.shape[1]))

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
data.graficarHsi(groundTruth)
data.graficarHsi(datosSalida)
'''