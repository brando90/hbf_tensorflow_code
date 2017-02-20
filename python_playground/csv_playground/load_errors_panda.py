import pandas as pd
import numpy as np

df = pd.read_csv('./tmp_errors/csv_errors_slurm_array_id5')
print(df)
print(df.as_matrix())
print(df.as_matrix().shape)
print(df.as_matrix()[:,0])
