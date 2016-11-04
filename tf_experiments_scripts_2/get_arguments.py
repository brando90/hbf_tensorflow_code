# Load the dictionary back from the pickle file.
import pickle
import namespaces as ns
import sys

# arg_dict = dict(arg).copy()
# arg_dict = get_remove_functions_from_dict(arg_dict)
# pickle.dump( arg_dict, open( "slurm-%s_%s.p"%(arg.slurm_jobid,arg.slurm_array_task_id) , "wb" ) )

if len(sys.argv) == 3:
    slurm_jobid = sys.argv[1]
    slurm_array_task_id = sys.argv[2]
else:
    slurm_jobid = 1
    slurm_array_task_id = 2

arg = pickle.load( open( "pickle-slurm-%s_%s.p"%(int(slurm_jobid)+int(slurm_array_task_id),slurm_array_task_id), "rb" ) )
print(arg)

arg_ns = ns.Namespace(arg)
#print('ns_args: ', ns.Namespace(arg) )
#print(arg_ns.dims)
