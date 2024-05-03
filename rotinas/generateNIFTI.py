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
       return
       
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
    for file in filesTmp:
        if 'nii.gz' in file and file[0] != 'o':
            dataTmp = os.path.join(dirTmp, file) 
            data = nib.load(dataTmp).get_fdata()  
    
    return data


#
# dirInput = '/mnt/dados/dataset_jpr_train/dataset_jpr_train'
dirInput = '/mnt/dados/dataset_jpr_test/dataset_jpr_test2'

dirTmp = '/mnt/dados/dataset_jpr_test/dataset_full/tmp'
dirOutput = '/mnt/dados/dataset_jpr_test/dataset_full'
# dirOutput2 = '/mnt/dados/dataset_jpr_train/dataset_4slices'
#
# qtdSlices=6
#

# errostxt = open('estudos_estranhos_tst.txt',"w")

if os.path.exists(dirTmp):
   shutil.rmtree(dirTmp)

if not os.path.exists(dirOutput):
    os.makedirs(dirOutput)
    
# if not os.path.exists(dirOutput2):
    # os.makedirs(dirOutput2)
    
    
# dirNbr = os.listdir(dirInput)  
# for Nbr in dirNbr:   
    # pathNbr = dirInput + '/' + Nbr
    
    # if os.path.isdir(pathNbr): 
        # dirID = os.listdir(pathNbr)
dirID = os.listdir(dirInput)
          
for ID in dirID:
    
    # pathID = dirInput + '/' + Nbr + '/' + ID 
    pathID = dirInput + '/' + ID                        
    
    outFile = os.path.join(dirOutput, f'{ID}full.nii.gz')   
    # outNewFile = os.path.join(dirOutput2, f'{ID}.nii.gz')
    
    if not os.path.exists(outFile):                
        data = dir2nii(pathID, outFile)
        print(f'{pathID} - {data.shape}')
        dataNifti = nib.Nifti1Image(data, affine=np.eye(4))    
        nib.save(dataNifti, outFile)
        print(f'{outFile}')
        
        if os.path.exists(dirTmp):
            shutil.rmtree(dirTmp)
                
            # else:
                # data = nib.load(outFile).get_fdata()
                
            # # if len(data.shape) == 4:
                # # # errostxt.write(f'{pathID} - {data.shape}\n')
                # # continue    
                
            # if not os.path.exists(outNewFile):  
            
                # if len(data.shape) == 4:
                    # a = np.mean(data[:,:,:,0], (0,1)).astype(int)
                # else:
                    # a = np.mean(data, (0,1)).astype(int)
                    
                # a0 = -999                
                # for i in range(len(a)-1,0,-1):    
                    # if (a[i] <= 0):
                        # a0 = a[i]
                    # else:
                        # last = i+1
                        # break
                        
                # nSlices = data.shape[2]
                # # normSlices = np.round(np.linspace(0, lastSlice+1, num=qtdSlices))
                # normSlices = np.array([0.60*last, 0.65*last, 0.7*last, 0.8*last])
                # normSlices = normSlices.astype(int)            
                
                # z=0
                # data3Vol = np.zeros((data.shape[0], data.shape[1], len(normSlices)))
                # for s in normSlices:
                    # if len(data.shape) == 4:
                        # dataNew[:,:,z] = data[:,:,s,0]
                        # data3Vol = data[:,:,[normSlices[s]-1,normSlices[s],normSlices[s]+1],0]
                    # else:
                        # dataNew[:,:,z] = data[:,:,s]
                        # data3Vol = data[:,:,[normSlices[s]-1,normSlices[s],normSlices[s]+1]]                        
                              
                                     
                    # for x, y in itertools.product(range(data3Vol.shape[0]), range(data3Vol.shape[1])):
                        # if data3Vol[x,y,1] >= 20:
                            # dataNew[x,y,z] = np.max(data3Vol[x,y,:]) 
                        # elif -20 <= data3Vol[x,y,1] < 20:
                            # dataNew[x,y,z] = np.min(data3Vol[x,y,:]) 
                        # elif data3Vol[x,y,1] < -25:
                            # dataNew[x,y,z] = -1024
    
                    # z+=1
               
                # dataNewNifti = nib.Nifti1Image(dataNew, affine=np.eye(4))    
                # nib.save(dataNewNifti, outNewFile)            
                # print(f'ls: {last} - {outNewFile} \n')
            
            
                    
        # errostxt.close()