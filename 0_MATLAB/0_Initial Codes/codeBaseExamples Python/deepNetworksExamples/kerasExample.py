from keras import models
from keras import layers
from keras import optimizers


model = models.Sequential() #MODELO DE LA RED
model.add(layers.Dense(32, activation='relu', input_shape=(784,)))  #CAPA DE ENTRADA y CAPA OCULTA con 32 Unidades ocultas
                                                                    #CAPAS OCULTAS
model.add(layers.Dense(10, activation='softmax'))                   #CAPA DE SALIDA
model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss='mse',metrics=['accuracy']) #FUNCION DE PERDIDA Y OPTIMMIZADOR

model.fit(input_tensor, target_tensor, batch_size=128, epochs=10)   #ENTRENAMIENTO

print("Done !!!!!!!!!!!!!!!!")