import os
import sys
import itertools
import numpy as np
import nibabel as nib
from scipy import ndimage
import itertools
from progress.bar import Bar


dirOutput = '/home/ubuntu/JPR/dataset_jpr_train/segmented_dataset_17slices'
dirInput = '/home/ubuntu/JPR/dataset_jpr_train/dataset_full_denoised'

k=2 

dirID = os.listdir(dirInput) 

if not os.path.exists(dirOutput):
    os.makedirs(dirOutput)

for ID in dirID:
    path = dirInput + '/' + ID    

    outFile = os.path.join(dirOutput, f'{ID.split("full")[0]}.nii.gz') 
    
    if not os.path.exists(outFile):  
        data = nib.load(path).get_fdata()
        print(f'{path} - {data.shape}')            
                
        if len(data.shape) == 4:
            a = np.mean(data[:,:,:,0], (0,1)).astype(int)
        else:
            a = np.mean(data, (0,1)).astype(int)
            
        a0 = float('inf')
        for i in range(len(a)-1,int(len(a)/2),-1):    
            if (a[i] <= -1000):
                a0 = a[i]
                last = i
            else:
                last = i
                break

        # normSlices = np.round(np.linspace(0, lastSlice+1, num=qtdSlices))
        # normSlices = np.array([0.35*last, 0.45*last, 0.55*last, 0.60*last, 0.65*last, 0.70*last, 0.75*last, 0.8*last, 0.85*last])
        # normSlices = np.array([0.35*last, 0.4*last, 0.45*last, 0.5*last, 0.55*last, 0.60*last, 0.65*last, 0.70*last, 0.75*last, 0.8*last, 0.85*last, 0.9*last, 0.95*last])
        normSlices = np.array([0.35*last, 0.4*last, 0.45*last, 0.5*last, 0.52*last, 0.55*last, 0.57*last, 0.60*last, 0.62*last, 0.65*last, 0.67*last, 0.70*last, 0.75*last, 0.8*last, 0.85*last, 0.9*last, 0.95*last])
        normSlices = normSlices.astype(int)           
        print(f'{normSlices} - ls:{last}')        
        
        dataNew = np.zeros((data.shape[0], data.shape[1], len(normSlices)))
        for z in range(len(normSlices)):
            if len(data.shape) == 4:                
                data3Vol = data[:,:,[normSlices[z]-1,normSlices[z],normSlices[z]+1],0]
            else:                
                data3Vol = data[:,:,[normSlices[z]-1,normSlices[z],normSlices[z]+1]]                   
                             
            rng = np.where(data3Vol[:,:,1] >= -20)                 
            for x, y in zip(rng[0],rng[1]):
                if data3Vol[x,y,1] >= 20:
                    dataNew[x,y,z] = np.max(data3Vol[x,y,:]) 
                # elif -20 <= data3Vol[x,y,1] < 20:
                    # dataNew[x,y,z] = np.min(data3Vol[x,y,:]) 
                elif data3Vol[x,y,1] < 20:
                    dataNew[x,y,z] = np.min(data3Vol[x,y,:])  

        dataNewNifti = nib.Nifti1Image(dataNew, affine=np.eye(4))    
        nib.save(dataNewNifti, outFile)   
        print(f'{outFile} sliced and denoised\n')                   


        # #DENOISE   
        
        # mask=np.zeros((dataNew.shape[0], dataNew.shape[1], dataNew.shape[2]))
        # maskC=np.copy(mask)
        # with Bar(f'Denoising', max=dataNew.shape[2]) as bar:  
            # for s in range(dataNew.shape[2]):
                # for x, y in itertools.product(range(k-1,dataNew.shape[0]-k,k), range(k-1,dataNew.shape[1]-k,k)):     

                    # gridX = range((x-k),(x+k))
                    # gridY = range((y-k),(y+k))

                    # knn = np.zeros(len(gridX)*len(gridY))
                    # ii=0
                    # for i, j in itertools.product(gridX, gridY):                
                        # knn[ii]=dataNew[i,j,s]
                        # ii+=1
                        
                    # res = np.all(knn>=-20)
                    # if res:
                        # for i, j in itertools.product(gridX, gridY):                
                            # mask[i,j,s] = 1

                # maskC[:,:,s] = ndimage.binary_closing(mask[:,:,s], iterations=35)
                # maskCf= maskC.astype('float')
                
                # bar.next()            

            # dataNewD=np.copy(dataNew)
            # dataNewD[np.where(maskCf==0)] = -1024 

            # dataNewNifti = nib.Nifti1Image(dataNew, affine=np.eye(4))    
            # nib.save(dataNewNifti, outFile)   
            # print(f'{outFile} sliced and denoised\n')