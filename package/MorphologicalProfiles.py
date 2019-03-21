#MP => Morphological Profiles using HSI
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology

class morphologicalProfiles:
    
    def __init__(self):
        pass

    def __str__(self):
        pass

    def thresholdsValues(self, data, num_t=10):
        thresholds = np.linspace(data.min(),data.max(),num_t+2)
        thresholds = thresholds[1:-1]
        return thresholds

    def thinning(self,imageChannel, threshold):
        image_binary = imageChannel < threshold
        out_thin =imageChannel*morphology.thin(image_binary)
        return out_thin

    def thickening(self,imageChannel, threshold):
        selem = morphology.disk(1)
        image_binary = imageChannel < threshold
        out_thic = imageChannel*morphology.dilation(image_binary, selem)
        return out_thic

    def EAP(self, imagen, num_thresholds=10): #Extended Attribute Profiles
        dataImage = imagen.copy()
        ImageEAP = np.zeros( ((2*num_thresholds+1)*dataImage.shape[0],dataImage.shape[1],dataImage.shape[2]) )
        k=0
        for i in range(dataImage.shape[0]):
            thresholds = self.thresholdsValues(dataImage[i], num_thresholds)
            for j in range(thresholds.shape[0]-1,-1,-1):  
                ImageEAP[k] = self.thickening(dataImage[i],thresholds[j])
                k += 1
            ImageEAP[k] = dataImage[i]
            k += 1
            for j in range(thresholds.shape[0]):  #thinning
                ImageEAP[k] = self.thinning(dataImage[i],thresholds[j])
                k += 1

        return ImageEAP
