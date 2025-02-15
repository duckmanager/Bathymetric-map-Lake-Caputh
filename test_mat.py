from pymatreader import read_mat
from pathlib import Path

data_dir = "data"
file_id = "20250122122110"
mat_file = Path(data_dir) / f"{file_id}.mat" 
mat_data = read_mat(str(mat_file))

input()