import os
from pathlib import Path

def read_files(root_dir: Path, Debug: bool=False):
    files1 = os.listdir(root_dir / '1')
    files2 = os.listdir(root_dir / '2')
    files3 = os.listdir(root_dir / '3')
    
    files1 = [root_dir / '1' / file for file in files1]
    files2 = [root_dir / '2' / file for file in files2]
    files3 = [root_dir / '3' / file for file in files3]
    
    return files1 + files2 + files3