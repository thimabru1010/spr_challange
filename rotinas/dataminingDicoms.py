import os
import sys
import numpy as np
import pandas as pd
import pickle
from glob import glob
import pydicom
import re
import csv

dirInput  = '/mnt/dados/dataset_jpr_train/dataset_jpr_train'

csvFile = open('infoDicoms.csv', 'w', newline='')
writer = csv.writer(csvFile)
writer.writerow(['ID', 'sexo', 'idade', 'qtdSlices', 'fabricante', 'kernel', 'linhas', 'colunas'])

dirNbr = os.listdir(dirInput) 
for Nbr in dirNbr:   
    pathNbr = dirInput + '/' + Nbr
    
    if os.path.isdir(pathNbr): 
        dirID = os.listdir(pathNbr)
        
        for ID in dirID:
            pathID = dirInput + '/' + Nbr + '/' + ID 
            
            dicoms = os.listdir(pathID)
            qtdSlices = len(dicoms)
            pathSample = dirInput + '/' + Nbr + '/' + ID + '/' + dicoms[0]
            
           
            ds = pydicom.dcmread(pathSample, force=True)    

            if len(ds.PatientAge) == 3:            
                age = ds.PatientAge[0:2]
            if len(ds.PatientAge) == 4:            
                age = ds.PatientAge[1:3]
            
            writer.writerow([ds.PatientID, ds.PatientSex, age, qtdSlices, ds.Manufacturer, ds.ConvolutionKernel, ds.Rows, ds.Columns])
            
csvFile.close()
