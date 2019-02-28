#PCA => Principal componet analysis using HSI
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class pcaAnalysis:
    
    def __init__(self,dataImagen):
        self.dataImagen = dataImagen

    def __str__(self):
        return f"Dimensiones Imagen:\t {self.dataImagen.shape}\n" 

    def funcionPrueba(self):
        pass