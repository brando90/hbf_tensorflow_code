import os

path_to_task_and_experiments = 'experiment_test_task'
#for (experiment_dir, _, potential_runs) in os.walk(path_to_experiments):
'''
As written, os path will go to every directory location recursively and print the contents of the directory, both the files and
the other directories it might contain.
More formally, each directory is a node in a graph and each node can have satellite data attached to it in the form of
files. Then the outdegree pointing to other nodes are other directories in the path. So when the search reaches a
specific node through a specific path, dirpth will be the path from top (root node) to current node and dirnames will be the
neighbour directories/nodes and the filenames will be the satellite data attached to that specific current node (pointed out by dirpath).
'''
# For each directory in the tree rooted at directory top (including top itself), it yields a 3-tuple (dirpath, dirnames, filenames).
for (dirpath, dirnames, filenames) in os.walk(top=path_to_task_and_experiments,topdown=True):
    #print('experiment_dir', experiment_dir)
    print('----')
    print('dirpath ', dirpath)
    print('dirnames ', dirnames)
    print('filenames ', filenames)

'''
For example run print:

brandomiranda~/Documents/MIT/MEng/hbf_tensorflow_code/tf_experiments_scripts/os_walk_test $ python os_walk_test.py
----
('dirpath ', 'experiment_test_task')
('dirnames ', ['test_NN_1', 'test_NN_2', 'test_NN_3'])
('filenames ', [])
----
('dirpath ', 'experiment_test_task/test_NN_1')
('dirnames ', [])
('filenames ', ['test1_NN1', 'test2_NN1', 'test3_NN1'])
----
('dirpath ', 'experiment_test_task/test_NN_2')
('dirnames ', [])
('filenames ', ['test1_NN2', 'test2_NN2', 'test3_NN2'])
----
('dirpath ', 'experiment_test_task/test_NN_3')
('dirnames ', [])
('filenames ', ['test1_NN3', 'test2_NN3', 'test3_NN3'])

'''
