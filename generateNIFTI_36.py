import os
import sys
import numpy as np
import nibabel as nib
import pydicom
import shutil

# Pacotes adicionais
# pip install python-gdcm
# conda install -c conda-forge dcm2niix



def dir2nii(dirSerie, outputFile):
    if os.path.exists(outputFile):
       return;
    print(dirSerie, outputFile)
    dirSaidaSerie = os.path.dirname(outputFile)  
    dirTmp = '%s/tmp' % dirSaidaSerie; 
    if not os.path.exists(dirTmp):
       os.makedirs(dirTmp);
       
    files = os.listdir(dirSerie);
    for file in files:
        os.system('gdcmconv -w "%s/%s" "%s/%s" > /dev/null 2>&1' % (dirSerie, file, dirTmp, file));

    output = dirTmp;    
    # os.system('dcm2nii -a -g -o "%s" "%s" > /dev/null 2>&1' % (output, output));
    os.system('dcm2niix -m y -z i -o "%s" "%s" > /dev/null 2>&1' % (output, output));    

    filesTmp = os.listdir(dirTmp);
    print(filesTmp)
    for file in filesTmp:
        print(file)
        print(file[0])
        if 'nii.gz' in file and file[0] != 'o':
            dataTmp = os.path.join(dirTmp, file) 
            data = nib.load(dataTmp).get_fdata()  
    
    return data


#
dirInput  = '/media/SSD2/IDOR/spr-head-ct-age-prediction-challenge/dataset_jpr_train'
dirTmp = '/media/SSD2/IDOR/spr-head-ct-age-prediction-challenge/dataset_jpr_train/dataset_36slices/tmp'
dirOutput = '/media/SSD2/IDOR/spr-head-ct-age-prediction-challenge/dataset_jpr_train/dataset_36slices'
#
qtdSlices=36
#
print('Starting...')
if not os.path.exists(dirOutput):
    os.makedirs(dirOutput)
    print('Creating output directory...')
    
dirNbr = os.listdir(dirInput)  
for Nbr in dirNbr:   
    pathNbr = dirInput + '/' + Nbr
    
    if os.path.isdir(pathNbr): 
        dirID = os.listdir(pathNbr)
        
        for ID in dirID:
            pathID = dirInput + '/' + Nbr + '/' + ID 
            
            # inFile = os.path.join(dirID, f'{ID}.nii.gz') 
            outFile = os.path.join(dirOutput, f'{ID}.nii.gz')            
            
            data = dir2nii(pathID, outFile)                   
            
            nSlices = data.shape[2]
            normSlices = np.round(np.linspace(0, nSlices-1, num=qtdSlices))            
            normSlices = normSlices.astype(int)            
            
            z=0
            dataNew = np.zeros((data.shape[0], data.shape[1], qtdSlices))
            for s in normSlices:                
                dataNew[:,:,z] = data[:,:,s]
                z+=1
           
            dataNewNifti = nib.Nifti1Image(dataNew, affine=np.eye(4))    
            nib.save(dataNewNifti, outFile)             
            shutil.rmtree(dirTmp);
            print(f'{outFile}')
