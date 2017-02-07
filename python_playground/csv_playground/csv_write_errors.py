import csv
import os
import pdb

def make_and_check_dir(path):
    '''
        tries to make dir/file, if it exists already does nothing else creates it.
    '''
    try:
        os.makedirs(path)
    except OSError:
        pass

path_root = './tmp_errors/'
make_and_check_dir(path_root)

#f = open('workfile', 'w')
#pdb.set_trace()
with open(path_root+'errors_csv_file.csv',mode='a+') as f_errors:
    #writer = csv.writer(f_errors)
    #writer = csv.DictWriter(f_errors,['val1', 'val2', 'val3'])
    writer = csv.DictWriter(f_errors)
    #writer.writeheader(['val1', 'val2', 'val3'])
    writer.writeheader()
    for i in range(15):
        a,b,c = i+0.1,i+0.2,i+0.3
        writer.writerow({'val1':a,'val2':b,'val3':c})

print('end')
