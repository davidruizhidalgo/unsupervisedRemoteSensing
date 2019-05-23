#Script utilizado para el desarrollo de pruebas en el codigo. 
import numpy as np
import glob
import matplotlib.pyplot as plt
from skimage.transform import resize

img = plt.imread("images/inception.png")
print(img.shape)

newSize = (300, 300)
img_resize = resize(img, newSize, anti_aliasing=True)
print(img_resize.shape)

numImg = len(glob.glob("images/*.png"))
images = np.zeros( (numImg,img_resize.shape[0],img_resize.shape[1],img_resize.shape[2]) )
print(images.shape)

for i in range(0,numImg):
        images[i] = img_resize

plt.figure(1)
plt.subplot(1,2,1)
plt.imshow(img)
plt.subplot(1,2,2)
plt.imshow(images[0])
plt.show()

####### EN DESARROLLO ##########################
# 1. Desarrollar una arquitectura de red selección de caracteristicas NO SUPERVISADA
        # k-PCA
        # Autoendoder
        # Grafo de autoencoders
        # CNN entrenada de forma no supervisada
# 2. Desarrollar un esquema de data augmentation => prepararDatos.py
# 4. Revisar documentacion reciente del estado del arte.