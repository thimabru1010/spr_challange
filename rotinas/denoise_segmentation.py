import os
import sys
import itertools
import numpy as np
import nibabel as nib
from scipy import ndimage
# from skimage.morphology import disk, binary_closing
from progress.bar import Bar


dirInput = '/home/ubuntu/JPR/dataset_jpr_test/dataset_full'
dirOutput = '/home/ubuntu/JPR/dataset_jpr_test/dataset_full_denoised'

k=4

if not os.path.exists(dirOutput):
    os.makedirs(dirOutput)


dirID = os.listdir(dirInput) 
for ID in dirID:
    path = dirInput + '/' + ID
    outFile = os.path.join(dirOutput, f'{ID}')
    
    if not os.path.exists(outFile):

        data = nib.load(path).get_fdata()
        if len(data.shape) == 4:
            data = data[:,:,:,0]
            
        data[(data<-20)]= -1024

        mask=np.zeros((data.shape[0], data.shape[1], data.shape[2]))
        maskC=np.zeros((data.shape[0], data.shape[1], data.shape[2]))

        with Bar(f'{ID}:', max=data.shape[2]) as bar:  
            for s in range(data.shape[2]):
                for x, y in itertools.product(range(0,data.shape[0]-k,k), range(0,data.shape[1]-k,k)):     

                    gridX = range((x-k),(x+k))
                    gridY = range((y-k),(y+k))

                    knn = np.zeros(len(gridX)*len(gridY))
                    ii=0
                    for i, j in itertools.product(gridX, gridY):                         
                        knn[ii]=data[i,j,s]
                        ii+=1
                        
                    res = (knn >= -10).sum() / (len(gridX)*len(gridY))    
                    if res>0.75:         
                         for i, j in itertools.product(gridX, gridY):                
                                mask[i,j,s] = 1

                maskC[:,:,s] = ndimage.binary_closing(mask[:,:,s], iterations=10*k)                

                bar.next()
                
        maskCf= maskC.astype('float')

        dataNew = nib.load(path).get_fdata()
        dataNew[np.where(maskCf==0)] = -1024 

        dataNewNifti = nib.Nifti1Image(dataNew, affine=np.eye(4))    
        nib.save(dataNewNifti, outFile)   
        print(f'{outFile}')