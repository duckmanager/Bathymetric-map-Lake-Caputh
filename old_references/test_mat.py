from pymatreader import read_mat
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

data_dir = "data"
file_id = "20250123121713"
mat_file = Path(data_dir) / f"{file_id}.mat" 
mat_data = read_mat(str(mat_file))

input()
plt.plot(mat_data['GPS']['GPS_Quality'])
plt.show()

input()