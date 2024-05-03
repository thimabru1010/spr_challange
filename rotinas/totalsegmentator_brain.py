import os

filename = '/mnt/dados/dataset_jpr_train/dataset_4slices/seg002469full.nii.gz'
dataFile = filename;
parentDir = os.path.dirname(filename);
os.system('TotalSegmentator -i "%s" -o "%s" -rs brain' % (dataFile, parentDir));