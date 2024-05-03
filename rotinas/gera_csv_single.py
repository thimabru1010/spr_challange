import pandas as pd
import numpy as np
from sklearn.utils import resample
import os
import nibabel as nib
from progress.bar import Bar


file = '/home/ubuntu/JPR/train_test.csv'
saveFile = '/home/ubuntu/JPR/train_test_groups.csv'
dirInput = '/home/ubuntu/JPR/dataset_jpr_train/dataset_full_denoised'

df = pd.read_csv(file)
age = df['Age']

group =[]
for i in range(0, len(age)):
    if (age[i] <= 45):
        group.append(0)
    elif (45 < age[i] <= 60):
        group.append(1)
    elif (age[i] > 60):
        group.append(2)
       
        
df['Group'] = group



ID, GRP, AGE = [],[],[]

dirID = os.listdir(dirInput) 

with Bar('Progresso:', max=len(dirID)) as bar:
    for IDD in dirID:
        path = dirInput + '/' + IDD    

        data = nib.load(path).get_fdata()
        nSlices = data.shape[2]
        
        name_vol = IDD.split("full")[0]
                
        csvi=df[df["StudyID"] == name_vol]
        if not csvi.empty:
            ind = csvi.index[0]
            for rep in range(nSlices):
                suf = f'_{rep}'                   
                ID.append(f'{df["StudyID"][ind]}{suf}')
                GRP.append(df['Group'][ind])
                AGE.append(df['Age'][ind])
        
        bar.next()

data = {'StudyID': ID,
        'Age': AGE,
        'Group': GRP}

# Create DataFrame
df2 = pd.DataFrame(data)
saveFile = '/home/ubuntu/JPR/train_test_groupsFULL.csv'
df2.to_csv(saveFile, index=False)
