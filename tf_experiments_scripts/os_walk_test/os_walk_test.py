import os

path_to_task_and_experiments = 'experiment_test_task'
#for (experiment_dir, _, potential_runs) in os.walk(path_to_experiments):
for (dirpath, dirnames, filenames) in os.walk(top=path_to_task_and_experiments,topdown=True):
    #print('experiment_dir', experiment_dir)
    print('----')
    print('dirpath ', dirpath)
    print('dirnames ', dirnames)
    print('filenames ', filenames)
