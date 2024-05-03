import os
import sys
import numpy as np
import nibabel as nib



dirOutput = '/home/ubuntu/JPR/dataset_jpr_train/segmented_dataset_36slices_single'
dirInput = '/home/ubuntu/JPR/dataset_jpr_train/segmented_dataset_36slices'

dirID = os.listdir(dirInput) 

if not os.path.exists(dirOutput):
    os.makedirs(dirOutput)

for ID in dirID:
    path = dirInput + '/' + ID    

    data = nib.load(path).get_fdata()
    nSlices = data.shape[2]
    
    for s in range(nSlices):      
        outFile = os.path.join(dirOutput, f'{ID.split(".nii.gz")[0]}_{s}.nii.gz') 
        
        if not os.path.exists(outFile):
            dataS = data[:,:,s]
            
            dataNewNifti = nib.Nifti1Image(dataS, affine=np.eye(4))    
            nib.save(dataNewNifti, outFile)   
            print(f'{outFile}')
            
            del dataS