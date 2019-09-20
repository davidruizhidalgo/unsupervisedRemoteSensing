#Morphological Profiles using HSI
#Extended Attribute Profiles => EAP
#Extended Extintion Profiles => EEP
import math
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import siamxt 

#np.seterr(divide='ignore', invalid='ignore')

class morphologicalProfiles:
    
    def __init__(self):
        pass

    def __str__(self):
        pass

    def connectivity(self, c_num):
        if c_num == 4:
            Bc = np.zeros((3,3),dtype = bool)
            Bc[1,:] = True
            Bc[:,1] = True
        else:
            Bc = np.ones((3,3),dtype = bool)
        return Bc

    def thresholdsValues(self, maxtree, num_t=10):  
        thresholds = np.linspace(maxtree.node_array[3,:].min(),maxtree.node_array[3,:].max(),num_t+2)
        thresholds = thresholds[1:-1]
        return thresholds

    def thinning(self,imageChannel, num_thresholds, indexT):  
        #Building the max-tree 
        Bc = np.ones((3,3),dtype = bool)
        mxt = siamxt.MaxTreeAlpha(imageChannel,Bc)
        #Thesholds
        thresholds = self.thresholdsValues(mxt, num_thresholds)
        #Applying an area-open filter
        mxt.areaOpen(thresholds[indexT])
        #Recovering the image 
        out_thin =  mxt.getImage()
        return out_thin

    def thickening(self,imageChannel, num_thresholds, indexT):  
        # Negating the image
        img_max = imageChannel.max()
        img_neg = img_max-imageChannel
        #Building the max-tree of the negated image, ==> min-tree
        Bc = np.ones((3,3),dtype = bool) #Conectivity 8
        mxt_neg = siamxt.MaxTreeAlpha(img_neg,Bc)
        #Thesholds
        thresholds = self.thresholdsValues(mxt_neg, num_thresholds)
        #Applying an area-open filter
        mxt_neg.areaOpen(thresholds[indexT])
        #Recovering the image 
        img_filtered =  mxt_neg.getImage()
        # Negating the image back
        out_thic = img_max -img_filtered
        return out_thic

    def EAP(self, imagen, num_thresholds=10): #Extended Attribute Profiles
        dataImage = imagen.copy()
        dataImage = (dataImage-dataImage.min())/(dataImage.max()-dataImage.min())
        dataImage = dataImage *1024
        Imagen = dataImage.astype(np.uint16)
        ImageEAP = np.zeros( ((2*num_thresholds+1)*dataImage.shape[0],dataImage.shape[1],dataImage.shape[2]) )

        k=0
        for i in range(Imagen.shape[0]):
            
            for j in range(num_thresholds-1,-1,-1): 
                ImageEAP[k] = self.thickening(Imagen[i],num_thresholds,j)  #thickening
                k += 1
            ImageEAP[k] = Imagen[i]
            k += 1
            for j in range(num_thresholds):  #thinning
                ImageEAP[k] = self.thinning(Imagen[i],num_thresholds,j)
                k += 1
        #NORMALIZAR DATOS DE SALIDA    
        mean = ImageEAP.mean(axis=0)
        ImageEAP -= mean
        std = ImageEAP.std(axis=0)
        ImageEAP /= std
        return ImageEAP

    def EEP(self, imagen, num_levels=7): #Extended Extintion Profiles
        dataImage = imagen.copy()
        dataImage = (dataImage-dataImage.min())/(dataImage.max()-dataImage.min())
        dataImage = dataImage *1024
        ImagenIn = dataImage.astype(np.uint16)
        # Parameters used to compute the extinction profile
        nextrema =  [int(2**jj) for jj in range(num_levels)][::-1]
        print("Nb. of extrema used to compute the profile:")
        print(nextrema)
        # Array to store the profile
        H,W = ImagenIn[0].shape
        Z = 2*len(nextrema)+1
        EEP = np.ones((ImagenIn.shape[0]*Z,H,W))
        for k in range(ImagenIn.shape[0]):
            ep = np.zeros((Z,H,W))
            imgChannel = ImagenIn[k]
            #Structuring element. connectivity-4 or connectivity-8
            Bc = self.connectivity(4)
            ######Min-tree Profile#########
            #Negating the image
            max_value = imgChannel.max()
            data_neg = (max_value - imgChannel)
            # Building the max-tree of the negated image, i.e. min-tree
            mxt = siamxt.MaxTreeAlpha(data_neg,Bc)
            # Area attribute extraction and computation of area extinction values
            area = mxt.node_array[3,:]
            Aext = mxt.computeExtinctionValues(area,"area")
            # Min-tree profile
            i = len(nextrema) - 1
            for n in nextrema:
                mxt2 = mxt.clone()
                mxt2.extinctionFilter(Aext,n)
                ep[i,:,:] = max_value - mxt2.getImage()
                i-=1
            # Putting the original image in the profile    
            i = len(nextrema)
            ep[i,:,:] = imgChannel
            i +=1
            ######Max-tree Profile#########
            #Building the max-tree
            mxt = siamxt.MaxTreeAlpha(imgChannel,Bc)
            # Area attribute extraction and computation of area extinction values
            area = mxt.node_array[3,:]
            Aext = mxt.computeExtinctionValues(area,"area")
            # Max-tree profile
            for n in nextrema:
                mxt2 = mxt.clone()
                mxt2.extinctionFilter(Aext,n)
                ep[i,:,:] = mxt2.getImage()
                i+=1
            #GENERAR EL EXTENDED EXTINTION PROFILE
            EEP[k*Z:(k+1)*Z,:,:] = ep[:,:,:]
        #NORMALIZAR DATOS DE SALIDA   
        #EEP = (EEP-EEP.min())/(EEP.max()-EEP.min()) 
        mean = EEP.mean(axis=0)
        EEP -= mean
        std = EEP.std(axis=0)
        EEP /= std
        return EEP
