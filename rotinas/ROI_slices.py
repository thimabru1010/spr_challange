import os
import sys
import numpy as np
import nibabel as nib
import pydicom
import csv
from progress.bar import Bar


dirInput = '/mnt/dados/dataset_jpr_train/segmented_dataset_9slices'

csvFile = open('medidas_SPR_9s.csv', 'w', newline='')
writer = csv.writer(csvFile)
header = ["ID", "mass_ratio", "fluid_ratio", "fm_ratio", "calc_ratio"]
writer.writerow(header)


dirID = os.listdir(dirInput) 

with Bar(f'Progress:', max=len(dirID)) as bar:
    for ID in dirID:
        path = dirInput + '/' + ID          
            
        data = nib.load(path).get_fdata()    
        nSlices = data.shape[2]

        mass = data[(20 < data) & (data < 100)].size
        fluid = data[(-15 < data) & (data < 15)].size
        calc = data[(100 < data) & (data < 300)].size
        bone = data[(300 < data)].size

        brain = mass+fluid+calc

        mass_ratio = mass/brain
        fluid_ratio = fluid/brain
        fm_ratio = fluid/mass
        calc_ratio = calc/brain

        ratios = [ID, mass_ratio, fluid_ratio, fm_ratio, calc_ratio]
        writer.writerow(ratios) 
        
        bar.next()