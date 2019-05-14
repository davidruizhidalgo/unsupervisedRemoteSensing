#Data Logger para el registro de datos de pruebas de entrenamiento y validacion
#savedataTrain() crea el logger con history generado por model.fit
import math
import numpy as np
import matplotlib.pyplot as plt
from io import open

class DataLogger:

    def __init__(self, fileName = 'mylogger'):
        nlogg = 'logger_'+fileName+'.txt'
        self.fichero = open(nlogg,'w')  

    def allocateVect(self, vector):
        for dato in vector:
            self.fichero.write(str(dato)+'\t')
        self.fichero.write('\n')

    def savedataTrain(self, history):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        #CREAR DATA LOGGER
        self.allocateVect(loss)
        self.allocateVect(val_loss)
        self.allocateVect(acc)
        self.allocateVect(val_acc)
    
    def savedataPerformance(self, OA, AA, kappa):
        OA_v = np.zeros((AA.shape[0]))
        OA_v[0] = OA
        kappa_v = np.zeros((AA.shape[0]))
        kappa_v[0] = kappa
        self.allocateVect(OA_v)                                 #Overall Accuracy 
        self.allocateVect(AA)                                   #Average Accuracy 
        self.allocateVect(kappa_v)                              #Kappa Coefficient

    def close(self):
        self.fichero.close()