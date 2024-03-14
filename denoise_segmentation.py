import os
import sys
import itertools
import numpy as np
import nibabel as nib
from scipy import ndimage
from progress.bar import Bar


dirInput = '/mnt/dados/dataset_jpr_train/dataset_36slices'
dirID = os.listdir(dirInput) 

k=3

for ID in dirID:
    path = dirInput + '/' + ID
    outFile = os.path.join(dirInput, f'seg{ID}')

    data = data = nib.load(path).get_fdata()
    mask=np.zeros((data.shape[0], data.shape[1], data.shape[2]))
    maskC=np.zeros((data.shape[0], data.shape[1], data.shape[2]))

    with Bar(f'{ID}:', max=data.shape[2]) as bar:  
        for s in range(data.shape[2]):
            for x, y in itertools.product(range(k-1,data.shape[0]-k,k), range(k-1,data.shape[1]-k,k)):     

                gridX = range((x-k),(x+k+1))
                gridY = range((y-k),(y+k+1))

                knn = np.zeros(len(gridX)*len(gridY))
                ii=0
                for i, j in itertools.product(gridX, gridY):                
                    knn[ii]=data[i,j,s]
                    ii+=1
                    
                res = np.all(knn>0) | np.all(knn<0)
                if res:
                     for i, j in itertools.product(gridX, gridY):                
                            mask[i,j,s] = 1

            maskC[:,:,s] = ndimage.binary_closing(mask[:,:,s], iterations=15)
            bar.next()
            
    maskCf= maskC.astype('float')

    dataNew=np.copy(data)
    dataNew[np.where(maskCf==0)] = -1024 

    dataNewNifti = nib.Nifti1Image(dataNew, affine=np.eye(4))    
    nib.save(dataNewNifti, outFile)   
    print(f'{outFile}')