import pandas as pd
import numpy as np

df = pd.read_csv('./tmp_errors/csv_errors_slurm_array_id5')
#print(df)
# print(type( list(df['train_error'])) )
# print(list(df['train_error']) )
# print(type(df['train_error'][0]) )
# print(np.float32( (df['train_error'][0]) ) )
print( df.to_dict() )
#print(df.as_matrix().shape)
#print(df.as_matrix()[:,0])
