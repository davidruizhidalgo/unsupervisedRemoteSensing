#Script utilizado para el desarrollo de pruebas en el codigo. 
####### EN DESARROLLO ##########################
# 1. Desarrollar una arquitectura de red selección de caracteristicas NO SUPERVISADA - Capa de Salida FINE-TUNNING
        # CNN Autoendoder and full conected network +/- 75% acc
            #Desempeño con EEP:  +/- 85% acc
            #SVM lineal and rbf  +/- 92% acc  and +/- 85% acc
            #Riemannian Geometry +/- 80% acc
        # Grafo de CNN autoencoders and full conected network +/- 80% acc
            #Desempeño con EEP:  +/- 96% acc
            #SVM lineal and rbf  +/- 98% acc  and +/- 90% acc
            #Riemannian Geometry +/- 99% acc
        # Evaluar modificaciones en la funcion de costo
        # Evaluar estrategias de concatenacion en BSCAE

# 2. Desarrollar un esquema de data augmentation => prepararDatos.py
# 4. Revisar documentacion reciente del estado del arte.

import matplotlib.pyplot as plt
import numpy as np