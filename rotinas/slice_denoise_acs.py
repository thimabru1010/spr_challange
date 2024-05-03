import os
import sys
import numpy as np
import nibabel as nib
import skimage


dirInput = '/home/ubuntu/JPR/dataset_jpr_train/dataset_full_denoised'
dirOutput = '/home/ubuntu/JPR/dataset_jpr_train/segmented_dataset_acs'


k=3 

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
        
        ix = int(0.5*data.shape[0])
        iy = int(0.5*data.shape[1])
        iz = int(0.6*last)


        dataNew = np.zeros((512, 512, 3))       

        #
        if len(data.shape) == 4:
            cx = data[ix,:,:,0]
            cy = data[:,iy,:,0]
            cz = data[:,:,iz,0]
            
        else:
            cx = data[ix,:,:]
            cy = data[:,iy,:]
            cz = data[:,:,iz]
            
        #  
        print(cx.shape, cy.shape, cz.shape)
        dataNew[:,:,0] = skimage.transform.resize(cx, (512, 512), order=3, preserve_range=True)              
        dataNew[:,:,1] = skimage.transform.resize(cy, (512, 512), order=3, preserve_range=True)
        if (cz.shape[0] != 512) or (cz.shape[1] != 512):
            dataNew[:,:,2] = skimage.transform.resize(cz, (512, 512), order=3, preserve_range=True)
        else:
            dataNew[:,:,2] = cz 

        dataNewNifti = nib.Nifti1Image(dataNew, affine=np.eye(4))    
        nib.save(dataNewNifti, outFile)   
        print(f'{outFile} sliced and denoised\n')