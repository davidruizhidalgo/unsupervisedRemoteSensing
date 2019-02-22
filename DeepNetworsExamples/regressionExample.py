#Predicting house prices: a regression example
from keras.datasets import boston_housing
from keras import models
from keras import layers

def build_model():
    # Because we will need to instantiate
    # the same model multiple time,
    # we use a function to construct it.
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

##ENCODING  DATA -> NORMALIZING THE DATA
'''Las trece clases tienen rangos diferentes de valores, lo cual puede ocacionar que el proceso de entrenamiento tenga dificultades al momente de realizar el ajuste. 
#debido a esto es una buena practica normalizar todas las caracteristicas en base a su media y desviación estandard. Así todas estaran centradas en cero y tendran
una unidad de desviacion estandard. '''
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

'''
Cuando existe un nuemro reducido de muestas, para validar el desempeño del entrenamiento se suele utilizar la estrategia denominada K-fold validation. En ella simplemente
se divide los datos en K particiones y entrenar K modelos, donde K-1 particiones de datos seran utulizadas para entrenas y la restante para evaluar el proceso de entrenamiento.
El puntaje de validacion esta dado por el promedio/media de los K valores de validacion obtenidos. 
'''

#MODEL DEFINITION
model = build_model()
# Train it on the entirety of the data.
model.fit(train_data, train_targets,
epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

print(test_mae_score)
print("done !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
