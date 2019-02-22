#CLASSIFYING MOVIE REVIEWS: BINARY CLASSIFICATION EXAMPLE
#DATA SET IMDB con 50K de reviews clasificados entre positivos y negativos

from keras.datasets import imdb 
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt

def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1. # set specific indices of results[i] to 1s
    return results


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)

# Our vectorized labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#MODELO RED NEURONAL
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

#ESTABLECER UN CONJUNTO DE VALIDACION PARA EL ENTRENAMIENTO
#x_val = x_train[:10000]
#partial_x_train = x_train[10000:]
#y_val = y_train[:10000]
#partial_y_train = y_train[10000:]
#ENTRENAR EL MODELO
#history = model.fit(partial_x_train,partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))

history = model.fit(x_train, y_train, epochs=15, batch_size=512)

#EL OBJETO history ES UN DICCIONARIO CON LOS DATOS DEL PROCESO DE ENTRENAMIENTO
##############graficar#########################
acc = history.history['acc']
#val_acc = history.history['val_acc']
loss = history.history['loss']
#val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
# "bo" is for "blue dot"
plt.figure(1) # new figure
plt.subplot(211) 
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
#plt.plot(epochs, val_loss, 'b', label='Validation loss')
#plt.title('Training and validation loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(212) 
acc_values = history.history['acc']
#val_acc_values = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
#plt.plot(epochs, val_acc, 'b', label='Validation acc')
#plt.title('Training and validation accuracy')
plt.title('Training accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#############fin graficar######################

#EVALUAR RED
results = model.evaluate(x_test, y_test)
print(results)


#UTILIZAR RED
prediction = model.predict(x_test)
print(prediction)
print("Done !!!!!!!!!!!!!!!!")